"""
Aspect Category Detection (ACD) - keyword matching (Section 3.5.3).

Reads the keyword lists in data/processed/aspect_keywords.csv (you can edit it),
checks every review, and sets:
  aspects_found    which of the 3 aspects were found, e.g. "service_reliability|app_performance"
  n_aspects        how many
  is_single_aspect 1 = exactly one aspect (used for aspect-level analysis)
                   0 = several or none  (only used for overall document-level sentiment)
  aspect           the aspect, when is_single_aspect = 1

Runs on the dictionary-mapped English text (step 4), once per review, so all three
language versions use the SAME aspect for the same review (fair comparison).

It also prints what you need to improve the keyword lists:
  - how many reviews have 0 / 1 / 2+ aspects
  - HOW MANY REVIEWS EACH KEYWORD MATCHES (delete noisy ones, find missing ones)
  - single-aspect reviews per telco in the dashboard window (this sets the equal
    test-sample size, so it replaces the 469 number)
  - example reviews with no aspect and with several aspects

Run from the project root:   python3 preprocessing/detect_aspects.py
Output: data/processed/aspect_detection.csv
"""

import csv
import re
from collections import Counter
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"
KEYWORDS_PATH = DATA / "aspect_keywords.csv"
INPUT_PATH = DATA / "4_dictionary_mapped_reviews.csv"
OUTPUT_PATH = DATA / "aspect_detection.csv"
WINDOW_START, WINDOW_END = "2025-09-01", "2026-09-30"

# FALLBACK RULE: if a review has NO aspect from the keyword lists but talks about the app
# itself ("best app", "good application"), call it app_performance. It is only used when
# nothing else was found, so it never turns a single-aspect review into a multi-aspect one.
# Set to False to switch it off and compare.
APP_FALLBACK_ON = True
APP_FALLBACK = re.compile(r"\b(?:app|apps|application|applications|aplikasi)\b")


def load_keywords():
    by_aspect = {}
    with open(KEYWORDS_PATH, encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        by_aspect.setdefault(row["aspect"].strip(), []).append(row["keyword"].strip().lower())
    return by_aspect


def compile_patterns(by_aspect):
    patterns = {}
    for aspect, words in by_aspect.items():
        words = sorted(set(w for w in words if w), key=len, reverse=True)
        patterns[aspect] = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")
    return patterns


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-keyword", action="store_true",
                        help="TEST: treat app/apps/application/aplikasi as a normal app_performance keyword "
                             "(a review that also has another aspect becomes multi-aspect). Saves a separate file.")
    args = parser.parse_args()
    global APP_FALLBACK_ON, OUTPUT_PATH
    if args.app_keyword:
        APP_FALLBACK_ON = False
        OUTPUT_PATH = DATA / "aspect_detection_appkeyword.csv"
        print("MODE: 'app' counts as an app_performance keyword (fallback rule OFF)\n")
    by_aspect = load_keywords()
    patterns = compile_patterns(by_aspect)
    print("Keywords per aspect:", {a: len(w) for a, w in by_aspect.items()})

    cols = ["review_id", "telco", "rating", "review_date", "source", "english_text"]
    df = pd.read_csv(INPUT_PATH, usecols=cols)
    df["english_text"] = df["english_text"].fillna("").astype(str).str.lower()

    hits = Counter()
    found_list = []
    via_fallback = []
    for text in df["english_text"]:
        found = []
        for aspect, pat in patterns.items():
            matches = set(pat.findall(text))
            if matches:
                found.append(aspect)
                for m in matches:
                    hits[(aspect, m)] += 1
        if args.app_keyword and APP_FALLBACK.search(text) and "app_performance" not in found:
            found.append("app_performance")
        if APP_FALLBACK_ON and not found and APP_FALLBACK.search(text):
            found = ["app_performance"]
            via_fallback.append(1)
        else:
            via_fallback.append(0)
        found_list.append(found)

    df["via_fallback"] = via_fallback
    df["aspects_found"] = ["|".join(f) for f in found_list]
    df["n_aspects"] = [len(f) for f in found_list]
    df["is_single_aspect"] = (df["n_aspects"] == 1).astype(int)
    df["aspect"] = [f[0] if len(f) == 1 else "" for f in found_list]
    df[["review_id", "telco", "aspects_found", "n_aspects", "is_single_aspect", "aspect", "via_fallback"]].to_csv(
        OUTPUT_PATH, index=False)

    total = len(df)
    print(f"\nReviews: {total}")
    for n, label in ((0, "no aspect"), (1, "ONE aspect (is_single_aspect = 1)"), (2, "two or more")):
        c = int((df["n_aspects"] == n).sum()) if n < 2 else int((df["n_aspects"] >= 2).sum())
        print(f"  {label:36s} {c:6d}  ({100 * c / total:.1f}%)")
    if APP_FALLBACK_ON:
        print(f"  (of these, {int(df['via_fallback'].sum())} got app_performance only from the 'app' fallback rule)")
    print("\nSingle-aspect reviews by aspect:")
    print(df[df["is_single_aspect"] == 1]["aspect"].value_counts().to_string())

    print("\nHits per keyword (reviews matched) - noisy or never-used keywords can be deleted:")
    for aspect in by_aspect:
        ranked = sorted(((m, c) for (a, m), c in hits.items() if a == aspect), key=lambda x: -x[1])
        never = [w for w in by_aspect[aspect] if (aspect, w) not in hits]
        print(f"\n  [{aspect}]  " + ", ".join(f"{m}={c}" for m, c in ranked[:25]))
        if never:
            print(f"     no matches at all: {', '.join(never)}")

    is_play = df["source"].str.contains("Google", case=False, na=False)
    date = pd.to_datetime(df["review_date"], errors="coerce")
    win = df[is_play & date.between(WINDOW_START, WINDOW_END)]
    print(f"\nDashboard window ({WINDOW_START} to {WINDOW_END}, Google Play) by telco:")
    table = win.groupby("telco").agg(window_reviews=("review_id", "count"),
                                     single_aspect=("is_single_aspect", "sum"))
    table["single_share_%"] = (100 * table["single_aspect"] / table["window_reviews"]).round(1)
    print(table.to_string())
    print(f"  -> equal test sample per telco is limited by the smallest single_aspect number")

    for title, mask in (("NO aspect found", df["n_aspects"] == 0),
                        ("TWO OR MORE aspects", df["n_aspects"] >= 2)):
        print(f"\nRandom reviews with {title} (read these: is a keyword missing or too loose?)")
        for _, r in df[mask].sample(n=min(10, int(mask.sum())), random_state=42).iterrows():
            print(f"  [{r['aspects_found'] or '-'}] {r['english_text'][:150]}")

    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
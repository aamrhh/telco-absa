"""
Suggests NEW keywords for the three aspects, from your own reviews.

Idea: reviews already matched to an aspect contain other words that show up much
more often there than in reviews overall (for customer service: "reply", "waiting",
"hours"...). If those words also appear in reviews that currently have NO aspect,
adding them would catch more reviews.

For each aspect it lists single words and two-word phrases that
  - are not already a keyword,
  - appear in at least --min-count reviews of that aspect,
  - are at least --min-lift times more common in that aspect's reviews than overall,
ranked by how many currently aspect-less reviews they would catch.

YOU decide. Read the examples, copy the good ones into data/processed/aspect_keywords.csv,
and run detect_aspects.py again. Skip words that fit several aspects.

Run from the project root:   python3 preprocessing/suggest_aspect_keywords.py
Output: data/processed/aspect_keyword_suggestions.csv  (also printed)
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_aspects import load_keywords, compile_patterns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = DATA / "4_dictionary_mapped_reviews.csv"
OUTPUT_PATH = DATA / "aspect_keyword_suggestions.csv"

STOP = set("""a an the and or but if so as at by for in of on to up out from with about into over than then
this that these those it its it's i me my we our you your he she they them their is am are was were be been being
do does did have has had will would can could should may might just very too really also more most much many some any
all one get got go going make made want need please pls lah la ya yg dan tak ada ni tu je ke la nak boleh jgk
app apps telco phone number""".split())
WORD = re.compile(r"[a-z][a-z']{2,}")


def grams(text):
    words = WORD.findall(text)
    out = {w for w in words if w not in STOP}
    out |= {f"{a} {b}" for a, b in zip(words, words[1:]) if a not in STOP and b not in STOP}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--min-lift", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    by_aspect = load_keywords()
    patterns = compile_patterns(by_aspect)
    existing = {w for words in by_aspect.values() for w in words}
    existing_re = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(existing, key=len, reverse=True)) + r")\b")

    texts = pd.read_csv(INPUT_PATH, usecols=["english_text"])["english_text"].fillna("").astype(str).str.lower().tolist()
    n = len(texts)
    total, none = Counter(), Counter()
    in_aspect = {a: Counter() for a in by_aspect}
    n_aspect = Counter()
    example = {}
    for text in texts:
        found = [a for a, p in patterns.items() if p.search(text)]
        g = grams(text)
        total.update(g)
        for a in found:
            in_aspect[a].update(g)
            n_aspect[a] += 1
        if not found:
            none.update(g)
            for x in g:
                example.setdefault(x, text[:140])
    print(f"{n} reviews. Reviews per aspect: {dict(n_aspect)}\n")

    rows = []
    for a in by_aspect:
        cands = []
        for g, c in in_aspect[a].items():
            if c < args.min_count or g in existing or existing_re.search(g):
                continue
            lift = (c / n_aspect[a]) / (total[g] / n)
            if lift >= args.min_lift and none[g] > 0:
                cands.append((none[g], g, c, round(lift, 1)))
        cands.sort(reverse=True)
        print(f"===== {a}: suggestions (new reviews caught = reviews with NO aspect now) =====")
        for new, g, c, lift in cands[:args.top]:
            print(f"  {g:28s} new={new:5d}  in_aspect={c:5d}  lift={lift:5.1f}   e.g. {example[g][:90]}")
            rows.append({"aspect": a, "suggestion": g, "new_reviews_caught": new,
                         "in_aspect_reviews": c, "lift": lift, "example": example[g]})
        print()
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
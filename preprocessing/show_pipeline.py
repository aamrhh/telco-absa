"""
show_pipeline.py  --  DEMO: watch one review go through every step of the pipeline.

   1. Segment   cut the RAW review into parts (punctuation, contrast words, joining words)
   2. Aspect    which aspect each part is about, and which keywords were found
   3. Clean     lower case, remove emoji / symbols
   4. Normalize spelling and short forms become one standard word      (shows what changed: old -> new)
   5. Dictionary meaning: English version and Malay version             (shows what changed: old -> new)

Run from the telco-absa folder:
    python3 preprocessing/show_pipeline.py                      <- 2 example reviews (invented, not from your data)
    python3 preprocessing/show_pipeline.py "your review text"   <- any text you type
    python3 preprocessing/show_pipeline.py --id 1234            <- a real review, by review_id
    python3 preprocessing/show_pipeline.py --random 3           <- 3 real reviews that are cut into 2+ parts

It uses the SAME functions as the real pipeline (segment_reviews.py, text_versions.py, manglish_normalizer.py,
dictionary_mapper.py), so what you show is what the pipeline really does.
"""
import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manglish_normalizer as nz                                      # noqa: E402,F401
import segment_reviews as sr                                          # noqa: E402
import text_versions as tv                                            # noqa: E402
from dictionary_mapper import load_dictionary, map_text               # noqa: E402

EXAMPLES = ["Aplikasi slow gila tapi customer service helpful. Line selalu putus la",
            "Cant login, otp x masuk2. Network kt umah teruk but the app UI is nice"]
LINE = "-" * 78


def changes(before: str, after: str) -> str:
    """Word-level difference, e.g. 'x -> tidak ; masuk2 -> masuk masuk'."""
    a, b = before.split(), after.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            out.append(f"{' '.join(a[i1:i2]) or '(nothing)'} -> {' '.join(b[j1:j2]) or '(removed)'}")
    return " ; ".join(out) if out else "(no change)"


def keywords_found(text: str, patterns: dict) -> str:
    low = text.lower()
    found = [f"{asp}: {m.group(0)}" for asp, pat in patterns.items() for m in pat.finditer(low)]
    return ", ".join(found[:4]) if found else "found by a rule (not a plain keyword)"


def explain(review: str, patterns, counts, lookup):
    """Run one review through every step. Returns one dict per part (used by the printout below AND by the dashboard)."""
    parts, _ = sr.segment(review, patterns, True, 2)
    cleaned = [tv.clean_text(p[0]) for p in parts]
    normed = tv.normalize_texts(cleaned, False, counts)
    out = []
    for (txt, asp, why, mixed), c, n in zip(parts, cleaned, normed):
        en, ms = map_text(n, lookup, "english"), map_text(n, lookup, "malay")
        two = "|" in str(asp)
        none = asp in ("none", "", None)
        out.append({"raw": txt, "cut": why, "aspect": "none" if none else str(asp), "two_aspects": two,
                    "keywords": "" if none else keywords_found(txt, patterns),
                    "cleaned": c, "normalized": n, "norm_changes": changes(c, n),
                    "english": en, "en_changes": changes(n, en), "malay": ms, "ms_changes": changes(n, ms)})
    return out


def show(review: str, patterns, counts, lookup):
    parts = explain(review, patterns, counts, lookup)
    print("\n" + "=" * 78)
    print("RAW REVIEW:", review)
    print("=" * 78)
    print(f"\nSTEP 1  SEGMENT   (raw text, before cleaning, because cleaning removes the dots and commas)")
    print(f"        {len(parts)} part(s):")
    for i, p in enumerate(parts, 1):
        print(f"        {i}. {p['raw']}      <- cut: {p['cut']}")
    print(f"\nSTEP 2  ASPECT")
    for i, p in enumerate(parts, 1):
        label = "none (general)" if p["aspect"] == "none" else p["aspect"].replace("|", " + ")
        print(f"        {i}. {label}{'  [two aspects: overall sentiment only]' if p['two_aspects'] else ''}")
        if p["aspect"] != "none":
            print(f"           because: {p['keywords']}")
    print(f"\nSTEP 3-5  CLEAN, NORMALIZE, DICTIONARY   (each part)")
    for i, p in enumerate(parts, 1):
        print(LINE)
        print(f"  part {i}  raw        : {p['raw']}")
        print(f"          cleaned    : {p['cleaned']}")
        print(f"          normalized : {p['normalized']}")
        print(f"             changed : {p['norm_changes']}")
        print(f"          English    : {p['english']}")
        print(f"             changed : {p['en_changes']}")
        print(f"          Malay      : {p['malay']}")
        print(f"             changed : {p['ms_changes']}")
    print(LINE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="a review to run through the pipeline")
    ap.add_argument("--id", help="a real review_id from 4_dictionary_mapped_reviews.csv")
    ap.add_argument("--random", type=int, help="show N real reviews that are cut into 2 or more parts")
    args = ap.parse_args()

    patterns = sr.compile_patterns(sr.filter_keywords(sr.load_keywords()))
    counts = tv.word_counts()
    lookup = load_dictionary()

    reviews = []
    if args.text:
        reviews = [args.text]
    elif args.id or args.random:
        df = pd.read_csv(sr.DATA / "4_dictionary_mapped_reviews.csv", dtype={"review_id": str}, usecols=["review_id", "review_text"])
        df["review_text"] = df["review_text"].fillna("").astype(str)
        if args.id:
            reviews = df.loc[df["review_id"] == args.id, "review_text"].tolist()
            if not reviews:
                sys.exit(f"review_id {args.id} not found")
        else:
            pool = df.sample(n=min(len(df), 400), random_state=None)
            reviews = [t for t in pool["review_text"] if len(sr.segment(t, patterns, True, 2)[0]) >= 2][:args.random]
    else:
        reviews = EXAMPLES
        print("(These 2 examples are invented, not from your data. Use --random 3 for real reviews.)")

    for r in reviews:
        show(r, patterns, counts, lookup)


if __name__ == "__main__":
    main()
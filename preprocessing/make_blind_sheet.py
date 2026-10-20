"""
make_blind_sheet.py  --  makes the sheet for your blind hand-labelling check.

It picks 100 + 100 + 100 parts from the TEST set (parts the SVM never trained on):
   100 that XLM-R called negative, 100 neutral, 100 positive.
The sheet shows you ONLY the text and the aspect. XLM-R's label is hidden in a separate key file.
The order is shuffled, so you cannot guess the bucket from the position.

Run:      python3 preprocessing/make_blind_sheet.py
Options:  --per-class 100   (how many parts per class)
          --by-aspect       (spread each class evenly over the 4 aspects: app_performance, service_reliability,
                             customer_service, none. Parts with two aspects are left out of this mode.)
          --force           (overwrite a sheet that you have already started labelling - careful!)

Needs:    data/processed/part_versions.csv   (made by run_part_experiment.py)
Writes:   data/processed/blind_label_sheet.csv   <- you fill the last column "my_label"
          data/processed/blind_key.csv           <- do NOT open this before you finish
"""
import argparse
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
CLASSES = ["negative", "neutral", "positive"]
SEED = 7
ASPECTS = ["app_performance", "service_reliability", "customer_service", "none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--by-aspect", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    old = DATA / "blind_label_sheet.csv"
    if old.exists() and not args.force:
        prev = pd.read_csv(old, encoding="utf-8-sig", dtype=str).fillna("")
        if (prev.get("my_label", pd.Series(dtype=str)).str.strip() != "").any():
            print("blind_label_sheet.csv already has labels in it. I did not touch it. Use --force only if you want to throw them away.")
            return

    pv = pd.read_csv(DATA / "part_versions.csv", encoding="utf-8-sig", dtype={"review_id": str, "part_id": str})
    te = pv[pv["split"] == "test"].copy()
    te["raw"] = te["raw"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    te = te[te["raw"] != ""]
    te = te.drop_duplicates("cleaned")          # the same text only once
    te = te.drop_duplicates("review_id")        # one part per review, so the sample is not dominated by long reviews
    print(f"Test parts to pick from: {len(te)}   {te['xlmr_label'].value_counts().to_dict()}")

    picks = []
    for c in CLASSES:
        pool = te[te["xlmr_label"] == c]
        if args.by_aspect:
            pool = pool[~pool["aspect"].fillna("none").str.contains(r"\|")]
            groups = [pool[pool["aspect"].fillna("none") == a] for a in ASPECTS]
            quota = [args.per_class // len(ASPECTS)] * len(ASPECTS)
            take = [min(q, len(g)) for q, g in zip(quota, groups)]
            for a, q, t in zip(ASPECTS, quota, take):
                if t < q:
                    print(f"  note: only {t} {c} parts in aspect {a} (wanted {q})")
            spare = args.per_class - sum(take)          # give the missing parts to the aspects that still have some
            while spare > 0 and any(len(g) > t for g, t in zip(groups, take)):
                for i, g in enumerate(groups):
                    if spare > 0 and len(g) > take[i]:
                        take[i] += 1
                        spare -= 1
            for g, t in zip(groups, take):
                picks.append(g.sample(n=t, random_state=SEED))
            continue
        n = min(args.per_class, len(pool))
        if n < args.per_class:
            print(f"  warning: only {n} {c} parts available")
        picks.append(pool.sample(n=n, random_state=SEED))
    s = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s.insert(0, "no", range(1, len(s) + 1))

    sheet = s[["no", "part_id", "aspect", "raw"]].rename(columns={"raw": "text"})
    sheet["my_label"] = ""
    sheet.to_csv(DATA / "blind_label_sheet.csv", index=False, encoding="utf-8-sig")
    s[["no", "part_id", "xlmr_label"]].to_csv(DATA / "blind_key.csv", index=False, encoding="utf-8-sig")

    print(f"\nWrote {len(sheet)} parts.")
    print("Aspect mix in the sheet:", s["aspect"].value_counts().to_dict())
    print("Class x aspect:\n", pd.crosstab(s["xlmr_label"], s["aspect"]))
    print(f"\nOpen {DATA / 'blind_label_sheet.csv'} in Excel. In the last column write one of:")
    print("   negative / neutral / positive     (or neg / neu / pos)")
    print("Judge the sentiment of the TEXT shown (about the aspect, if the text has one). Save as CSV when done.")
    print("Do not open blind_key.csv until you have finished.")


if __name__ == "__main__":
    main()
"""
run_baselines.py  --  the "simple tools" for comparison: VADER (and Malaya, if it works on your Mac).

It labels the SAME test parts that the SVMs were tested on, on the RAW text, and scores them
   (a) against XLM-R's labels (the same yardstick as the SVM table), and
   (b) against YOUR blind labels, if blind_label_sheet.csv is filled in.

Run:       python3 preprocessing/run_baselines.py
Options:   --malaya-limit 3000   (how many test parts Malaya labels; Malaya is slow. 0 = skip Malaya)
Install:   pip install vaderSentiment        (once, in your venv)

Needs:     data/processed/part_versions.csv   (made by run_part_experiment.py)
Writes:    data/processed/baseline_results.csv
VADER: compound score >= 0.05 -> positive, <= -0.05 -> negative, otherwise neutral (the usual rule). English words only.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
CLASSES = ["negative", "neutral", "positive"]
LABELS = {"negative": "negative", "neg": "negative", "n": "negative", "-": "negative",
          "neutral": "neutral", "neu": "neutral", "0": "neutral",
          "positive": "positive", "pos": "positive", "p": "positive", "+": "positive"}


def vader_labels(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = an.polarity_scores(str(t))["compound"]
        out.append("positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")
    return out


def malaya_labels(texts):
    """Malaya's sentiment model. The Malaya functions change between versions, so this tries and tells you if it fails."""
    import malaya
    model = malaya.sentiment.huggingface()
    out = []
    for i in range(0, len(texts), 32):
        res = model.predict(list(texts[i:i + 32]))
        for r in res:
            r = str(r.get("label") if isinstance(r, dict) else r).lower()
            out.append("positive" if r.startswith("pos") else "negative" if r.startswith("neg") else "neutral")
        print(f"  Malaya labelled {len(out)}/{len(texts)}", end="\r")
    print()
    return out


def score(gold, pred, who, against, rows):
    f = f1_score(gold, pred, average=None, labels=CLASSES, zero_division=0)
    row = {"Tool": who, "Scored against": against, "Parts": len(gold),
           "Accuracy %": round(accuracy_score(gold, pred) * 100, 1),
           "Macro-F1 %": round(f1_score(gold, pred, average="macro", labels=CLASSES, zero_division=0) * 100, 1),
           "F1 negative %": round(f[0] * 100, 1), "F1 neutral %": round(f[1] * 100, 1), "F1 positive %": round(f[2] * 100, 1)}
    rows.append(row)
    print(f"  {who:28s} vs {against:14s} accuracy {row['Accuracy %']:5.1f}%  macro-F1 {row['Macro-F1 %']:5.1f}%  "
          f"(neg {row['F1 negative %']:.0f} / neu {row['F1 neutral %']:.0f} / pos {row['F1 positive %']:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malaya-limit", type=int, default=3000)
    args = ap.parse_args()

    pv = pd.read_csv(DATA / "part_versions.csv", encoding="utf-8-sig", dtype={"part_id": str, "review_id": str})
    te = pv[pv["split"] == "test"].copy()
    te["raw"] = te["raw"].fillna("").astype(str)
    print(f"{len(te)} test parts. Always-negative baseline: {te['xlmr_label'].value_counts().max() / len(te) * 100:.1f}%\n")

    rows = []
    print("Against XLM-R's labels (same test parts as the SVM table):")
    te["vader"] = vader_labels(te["raw"])
    score(te["xlmr_label"], te["vader"], "VADER", "XLM-R labels", rows)
    mal = None
    if args.malaya_limit:
        sub = te.sample(n=min(args.malaya_limit, len(te)), random_state=42)
        try:
            sub = sub.assign(malaya=malaya_labels(sub["raw"].tolist()))
            score(sub["xlmr_label"], sub["malaya"], f"Malaya ({len(sub)} parts)", "XLM-R labels", rows)
            mal = sub.set_index("part_id")["malaya"]
        except Exception as e:                                                  # noqa: BLE001
            print(f"  Malaya did not run: {type(e).__name__}: {str(e)[:300]}")
            print("  (send me this message and I will fix it; VADER is still scored.)")

    sheet = DATA / "blind_label_sheet.csv"
    if sheet.exists():
        sh = pd.read_csv(sheet, encoding="utf-8-sig", dtype=str).fillna("")
        sh["gold"] = sh["my_label"].str.strip().str.lower().map(LABELS)
        sh = sh.dropna(subset=["gold"])
        if len(sh):
            print(f"\nAgainst YOUR blind labels ({len(sh)} parts):")
            key = pd.read_csv(DATA / "blind_key.csv", encoding="utf-8-sig", dtype=str)
            sh = sh.merge(key[["part_id", "xlmr_label"]], on="part_id")
            sh["vader"] = vader_labels(sh["text"])
            score(sh["gold"], sh["xlmr_label"], "XLM-R", "your labels", rows)
            score(sh["gold"], sh["vader"], "VADER", "your labels", rows)
            if args.malaya_limit and mal is not None:
                try:
                    sh["malaya"] = malaya_labels(sh["text"].tolist())
                    score(sh["gold"], sh["malaya"], "Malaya", "your labels", rows)
                except Exception as e:                                          # noqa: BLE001
                    print(f"  Malaya did not run on your sheet: {type(e).__name__}: {str(e)[:200]}")
    pd.DataFrame(rows).to_csv(DATA / "baseline_results.csv", index=False)
    print(f"\nSaved baseline_results.csv in {DATA}")


if __name__ == "__main__":
    main()
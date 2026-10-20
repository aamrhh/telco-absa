"""
run_same420.py  --  PAGE 4: the cleanest comparison. The SAME 420 texts, three different label sources.

The SVM is trained on your 420 parts, with the labels coming from:
    (a) XLM-R       (b) DistilBERT       (c) YOU (your hand labels)
and it is always TESTED against YOUR labels, on parts it has never seen (cross-validation).
Because the texts and the folds are the same, the ONLY thing that changes is who made the training labels.

Run:     python3 preprocessing/run_same420.py
Needs:   review_segments_raw.csv, part_xlmr_labels.csv, part_distil_labels.csv, blind_label_sheet.csv (filled in)
Writes:  data/processed/same420.xlsx  and  same420.csv

Cross-validation: 5 folds, repeated 10 times (50 rounds). Each round trains on about 336 parts and tests on the other 84.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manglish_normalizer as nz                                      # noqa: E402
import text_versions as tv                                            # noqa: E402
from dictionary_mapper import load_dictionary, map_text               # noqa: E402

DATA = tv.DATA
CLASSES = ["negative", "neutral", "positive"]
LABELS = {"negative": "negative", "neg": "negative", "n": "negative", "-": "negative",
          "neutral": "neutral", "neu": "neutral", "0": "neutral",
          "positive": "positive", "pos": "positive", "p": "positive", "+": "positive"}
VERSIONS = [("Raw", "raw"), ("Cleaned only", "cleaned"), ("Normalized (after cleaning)", "norm"),
            ("English dictionary", "dict_en"), ("Malay dictionary", "dict_ms")]
SOURCES = [("XLM-R labels", "xlmr_label"), ("DistilBERT labels", "distil_label"), ("My labels", "gold")]


def make_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                         LinearSVC(C=1.0, class_weight="balanced", random_state=42))


def main():
    seg = pd.read_csv(DATA / "review_segments_raw.csv", dtype={"review_id": str})
    seg["part_id"] = seg["review_id"] + "_" + seg["segment_id"].astype(str)
    seg["segment_text"] = seg["segment_text"].fillna("").astype(str)
    xl = pd.read_csv(DATA / "part_xlmr_labels.csv", dtype={"part_id": str})[["part_id", "xlmr_label"]]
    di = pd.read_csv(DATA / "part_distil_labels.csv", dtype={"part_id": str})[["part_id", "distil_label"]]
    sh = pd.read_csv(DATA / "blind_label_sheet.csv", encoding="utf-8-sig", dtype=str).fillna("")
    sh["gold"] = sh["my_label"].str.strip().str.lower().map(LABELS)
    sh = sh.dropna(subset=["gold"])
    d = seg.merge(sh[["part_id", "gold"]], on="part_id").merge(xl, on="part_id", how="left").merge(di, on="part_id", how="left")
    d = d.dropna(subset=["xlmr_label", "distil_label"]).reset_index(drop=True)
    print(f"{len(d)} parts.  My labels: {d['gold'].value_counts().to_dict()}")
    print(f"  XLM-R labels      : {d['xlmr_label'].value_counts().to_dict()}   agree with me {np.mean(d['xlmr_label'] == d['gold']) * 100:.1f}%")
    print(f"  DistilBERT labels : {d['distil_label'].value_counts().to_dict()}   agree with me {np.mean(d['distil_label'] == d['gold']) * 100:.1f}%")

    print("\nMaking the text versions ...")
    counts = tv.word_counts()
    full = load_dictionary()
    d["raw"] = d["segment_text"]
    d["cleaned"] = [tv.clean_text(t) for t in d["raw"]]
    d["norm"] = tv.normalize_texts(d["cleaned"], False, counts)
    d["dict_en"] = [map_text(t, full, "english") for t in d["norm"]]
    d["dict_ms"] = [map_text(t, full, "malay") for t in d["norm"]]
    nz.USE_SPELLCHECKER = False

    y = d["gold"].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)     # the SAME folds for every source and version
    splits = list(cv.split(d["raw"], y))
    rows, table = [], {label: {"Text version": label} for label, _ in VERSIONS}
    print("Training (5 folds x 10 repeats) and testing against YOUR labels ...")
    for label, col in VERSIONS:
        x = d[col].fillna("").to_numpy()
        for sname, scol in SOURCES:
            ytrain = d[scol].to_numpy()
            acc, mf = [], []
            for tr, te in splits:
                if len(set(ytrain[tr])) < 3:
                    continue
                p = make_model().fit(x[tr], ytrain[tr]).predict(x[te])
                acc.append(accuracy_score(y[te], p))
                mf.append(f1_score(y[te], p, average="macro", labels=CLASSES, zero_division=0))
            table[label][f"{sname}: % right"] = round(np.mean(acc) * 100, 1)
            table[label][f"{sname}: fair score"] = round(np.mean(mf) * 100, 1)
            rows.append({"Text version": label, "Training labels": sname, "% right": round(np.mean(acc) * 100, 1),
                         "+/-": round(np.std(acc) * 100, 1), "Fair score": round(np.mean(mf) * 100, 1),
                         "+/- (fair)": round(np.std(mf) * 100, 1)})
            print(f"  {label:28s} trained on {sname:18s}: right {np.mean(acc) * 100:5.1f}% (+/- {np.std(acc) * 100:.1f})   "
                  f"fair {np.mean(mf) * 100:5.1f}%")
    tab = pd.DataFrame(list(table.values()))
    base = d["gold"].value_counts().max() / len(d) * 100
    print(f"\n=== Same {len(d)} texts, three label sources, always tested against MY labels ===")
    print(tab.to_string(index=False))
    print(f"\n(Always-negative baseline: {base:.1f}% right.)")
    pd.DataFrame(rows).to_csv(DATA / "same420.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "same420.xlsx") as xl_:
            tab.to_excel(xl_, sheet_name="Same 420 texts", index=False)
            pd.DataFrame(rows).to_excel(xl_, sheet_name="With spread", index=False)
            pd.DataFrame({"How to read": [
                "The same 420 texts and the same cross-validation folds are used for every row and column.",
                "Only the TRAINING labels change: XLM-R, DistilBERT, or mine. The test labels are always mine.",
                "% right = same as my label. Fair score = macro-F1 (average over negative, neutral, positive).",
                f"Always saying the biggest class would be right {base:.1f}% of the time on these parts.",
                "+/- (sheet 'With spread') is how much the score moves from round to round."]}).to_excel(
                xl_, sheet_name="Notes", index=False)
    except Exception as e:                                                       # noqa: BLE001
        print("Could not write the xlsx:", e)
    print(f"\nSaved same420.xlsx and same420.csv in {DATA}")


if __name__ == "__main__":
    main()
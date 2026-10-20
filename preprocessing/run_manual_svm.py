"""
run_manual_svm.py  --  "page 3": train AND test the SVM on YOUR OWN labels only. No auto-labeler at all.

You only have 420 hand labels, so one train/test split would be very noisy. Instead:
   5-fold cross-validation, repeated 10 times (each time the 420 parts are shuffled differently).
   In every round, the SVM trains on about 336 of your parts and is tested on the other 84 it has never seen.
   The number shown is the average over all 50 rounds, with the spread (+/-).
It is done for every text version: Raw, Cleaned only, Normalized, English dictionary, Malay dictionary.

Run:     python3 preprocessing/run_manual_svm.py
Needs:   review_segments_raw.csv, blind_label_sheet.csv (filled in)
Writes:  data/processed/manual_svm.xlsx  and  manual_svm.csv

Important: 420 parts is a SMALL training set (the other SVMs trained on about 20,000). Expect lower scores. This page
answers: "how far can I get with only my own labels?"
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


def make_model():
    # min_df=1 because the training set is tiny; class_weight="balanced" because neutral is rare in your labels
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                         LinearSVC(C=1.0, class_weight="balanced", random_state=42))


def main():
    seg = pd.read_csv(DATA / "review_segments_raw.csv", dtype={"review_id": str})
    seg["part_id"] = seg["review_id"] + "_" + seg["segment_id"].astype(str)
    seg["segment_text"] = seg["segment_text"].fillna("").astype(str)
    sh = pd.read_csv(DATA / "blind_label_sheet.csv", encoding="utf-8-sig", dtype=str).fillna("")
    sh["gold"] = sh["my_label"].str.strip().str.lower().map(LABELS)
    sh = sh.dropna(subset=["gold"])
    d = seg.merge(sh[["part_id", "gold"]], on="part_id").reset_index(drop=True)
    print(f"{len(d)} of your {len(sh)} labelled parts found.  Your labels: {d['gold'].value_counts().to_dict()}")
    base = d["gold"].value_counts().max() / len(d) * 100
    print(f"Always saying '{d['gold'].value_counts().idxmax()}' would be right {base:.1f}% of the time on these parts.\n")

    print("Making the text versions ...")
    counts = tv.word_counts()
    full = load_dictionary()
    d["raw"] = d["segment_text"]
    d["cleaned"] = [tv.clean_text(t) for t in d["raw"]]
    d["norm"] = tv.normalize_texts(d["cleaned"], False, counts)
    d["dict_en"] = [map_text(t, full, "english") for t in d["norm"]]
    d["dict_ms"] = [map_text(t, full, "malay") for t in d["norm"]]
    nz.USE_SPELLCHECKER = False

    y = d["gold"].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    rows = []
    print("Training and testing (5 folds x 10 repeats = 50 rounds per version) ...")
    for label, col in VERSIONS:
        x = d[col].fillna("").to_numpy()
        acc, mf, f1c = [], [], []
        for tr, te in cv.split(x, y):
            m = make_model().fit(x[tr], y[tr])
            p = m.predict(x[te])
            acc.append(accuracy_score(y[te], p))
            mf.append(f1_score(y[te], p, average="macro", labels=CLASSES, zero_division=0))
            f1c.append(f1_score(y[te], p, average=None, labels=CLASSES, zero_division=0))
        f1c = np.mean(f1c, axis=0)
        rows.append({"Text version": label, "% right": round(np.mean(acc) * 100, 1), "+/-": round(np.std(acc) * 100, 1),
                     "Fair score": round(np.mean(mf) * 100, 1), "+/- (fair)": round(np.std(mf) * 100, 1),
                     "F1 negative": round(f1c[0] * 100, 1), "F1 neutral": round(f1c[1] * 100, 1), "F1 positive": round(f1c[2] * 100, 1)})
        print(f"  {label:28s}: right {rows[-1]['% right']:5.1f}% (+/- {rows[-1]['+/-']:.1f})   "
              f"fair score {rows[-1]['Fair score']:5.1f}% (+/- {rows[-1]['+/- (fair)']:.1f})   "
              f"(neg {f1c[0] * 100:.0f} / neu {f1c[1] * 100:.0f} / pos {f1c[2] * 100:.0f})")
    tab = pd.DataFrame(rows)
    print(f"\n=== SVM trained AND tested on your own labels ({len(d)} parts, cross-validation) ===")
    print(tab.to_string(index=False))
    print(f"\n(Always-{d['gold'].value_counts().idxmax()} baseline: {base:.1f}% right.)")
    tab.to_csv(DATA / "manual_svm.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "manual_svm.xlsx") as xl:
            tab.to_excel(xl, sheet_name="Manual SVM", index=False)
            pd.DataFrame({"How to read": [
                f"The SVM trains only on your own {len(d)} labels. No auto-labeler is used.",
                "5-fold cross-validation repeated 10 times: each round trains on about 80% of your parts and tests on the other 20%.",
                "% right = how many the SVM got the same as you. Fair score = macro-F1, the average over negative, neutral, positive.",
                "+/- is how much the number moves from round to round. Small differences between versions are within luck.",
                f"Always saying the biggest class would be right {base:.1f}% of the time on these parts."]}).to_excel(
                xl, sheet_name="Notes", index=False)
    except Exception as e:                                                       # noqa: BLE001
        print("Could not write the xlsx:", e)
    print(f"\nSaved manual_svm.xlsx and manual_svm.csv in {DATA}")


if __name__ == "__main__":
    main()
    
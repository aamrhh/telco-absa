"""
build_dashboard_data.py  --  prepare everything the dashboard needs. Run it ONCE (about 5 to 15 minutes), and again if you change the labels.

The dashboard uses ONE model: the SVM trained on your own 420 labels (no auto-labeler). This script:
  1. trains that SVM on ALL 420 labels, once per language version
        Manglish = normalized text (original words kept)   English = English dictionary text   Malay = Malay dictionary text
  2. uses each SVM to predict the sentiment of ALL parts (92,242)
  3. measures how good each SVM is (cross-validation: 5 folds x 10 repeats, the same as run_manual_svm.py)

Run from the telco-absa folder:
    python3 preprocessing/build_dashboard_data.py
    (the dashboard shows Google Play reviews from 1 Sep 2025 to 30 Sep 2026; change with --from, --to, --source)

Needs:   review_segments_raw.csv, 4_dictionary_mapped_reviews.csv, blind_label_sheet.csv (filled in), 2_tokenized_reviews.csv
Writes:  data/dashboard/parts_dashboard.csv, svm_manglish.joblib, svm_english.joblib, svm_malay.joblib,
         word_counts.joblib, performance.json
Note:    the 420 parts that the SVM was trained on are marked in_420 = True (their predictions are not a fair test).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manglish_normalizer as nz                                      # noqa: E402
import text_versions as tv                                            # noqa: E402
from dictionary_mapper import load_dictionary, map_text               # noqa: E402

DATA = tv.DATA
OUT = DATA.parent / "dashboard"
CLASSES = ["negative", "neutral", "positive"]
LABELS = {"negative": "negative", "neg": "negative", "n": "negative", "-": "negative",
          "neutral": "neutral", "neu": "neutral", "0": "neutral",
          "positive": "positive", "pos": "positive", "p": "positive", "+": "positive"}
VERSIONS = {"manglish": "norm", "english": "dict_en", "malay": "dict_ms"}      # dashboard name -> text version column


def make_model():
    # same settings as run_manual_svm.py (page 3)
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                         LinearSVC(C=1.0, class_weight="balanced", random_state=42))


def add_versions(df, counts, full):
    df["raw"] = df["segment_text"]
    df["cleaned"] = [tv.clean_text(t) for t in df["raw"]]
    df["norm"] = tv.normalize_texts(df["cleaned"], False, counts)
    df["dict_en"] = [map_text(t, full, "english") for t in df["norm"]]
    df["dict_ms"] = [map_text(t, full, "malay") for t in df["norm"]]
    nz.USE_SPELLCHECKER = False
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2025-09-01", help="first review date shown in the dashboard")
    ap.add_argument("--to", dest="end", default="2026-09-30", help="last review date shown in the dashboard")
    ap.add_argument("--source", default="Google Play", help='which source the dashboard shows ("all" for every source)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(DATA / "review_segments_raw.csv", dtype={"review_id": str})
    seg["part_id"] = seg["review_id"] + "_" + seg["segment_id"].astype(str)
    seg["segment_text"] = seg["segment_text"].fillna("").astype(str)
    seg["aspect"] = seg["aspect"].fillna("none").astype(str)
    meta = pd.read_csv(DATA / "4_dictionary_mapped_reviews.csv", usecols=["review_id", "telco", "review_date", "source"],
                       dtype={"review_id": str})
    sh = pd.read_csv(DATA / "blind_label_sheet.csv", encoding="utf-8-sig", dtype=str).fillna("")
    sh["gold"] = sh["my_label"].str.strip().str.lower().map(LABELS)
    sh = sh.dropna(subset=["gold"])
    print(f"{len(seg)} parts, {len(sh)} hand-labelled parts. Your labels: {sh['gold'].value_counts().to_dict()}")

    print("\nMaking the text versions for ALL parts (this is the slow step) ...")
    counts = tv.word_counts()
    full = load_dictionary()
    seg = add_versions(seg, counts, full)
    joblib.dump(dict(counts), OUT / "word_counts.joblib")

    lab = seg.merge(sh[["part_id", "gold"]], on="part_id").reset_index(drop=True)
    y = lab["gold"].to_numpy()
    base = float(pd.Series(y).value_counts().max() / len(y) * 100)
    print(f"\nTraining on {len(lab)} labelled parts: {pd.Series(y).value_counts().to_dict()}   (always-biggest-class: {base:.1f}%)")

    perf = {"n_labels": int(len(lab)), "class_counts": {c: int((y == c).sum()) for c in CLASSES},
            "baseline_accuracy": round(base, 1), "cv": "5 folds x 10 repeats", "versions": {}}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    splits = list(cv.split(lab["raw"], y))
    for name, col in VERSIONS.items():
        x = lab[col].fillna("").to_numpy()
        acc, mf, f1c, cm = [], [], [], np.zeros((3, 3))
        for tr, te in splits:
            p = make_model().fit(x[tr], y[tr]).predict(x[te])
            acc.append(accuracy_score(y[te], p))
            mf.append(f1_score(y[te], p, average="macro", labels=CLASSES, zero_division=0))
            f1c.append(f1_score(y[te], p, average=None, labels=CLASSES, zero_division=0))
            cm += confusion_matrix(y[te], p, labels=CLASSES)
        f1c = np.mean(f1c, axis=0)
        perf["versions"][name] = {
            "text_column": col, "accuracy": round(np.mean(acc) * 100, 1), "accuracy_sd": round(np.std(acc) * 100, 1),
            "macro_f1": round(np.mean(mf) * 100, 1), "macro_f1_sd": round(np.std(mf) * 100, 1),
            "f1": {c: float(round(v * 100, 1)) for c, v in zip(CLASSES, f1c)},
            "confusion": (cm / len(splits) * 5).round(1).tolist()}          # rows = true, cols = predicted; adds up to about 420
        print(f"  {name:9s}: accuracy {perf['versions'][name]['accuracy']:5.1f}%   macro-F1 {perf['versions'][name]['macro_f1']:5.1f}%   "
              f"F1 {perf['versions'][name]['f1']}")
        model = make_model().fit(x, y)                                       # the final model: ALL 420 labels
        joblib.dump(model, OUT / f"svm_{name}.joblib")
        seg[f"pred_{name}"] = model.predict(seg[col].fillna(""))
    (OUT / "performance.json").write_text(json.dumps(perf, indent=2))

    seg = seg.merge(meta, on="review_id", how="left")
    seg["month"] = pd.to_datetime(seg["review_date"], errors="coerce").dt.to_period("M").astype(str)
    seg = seg.merge(sh[["part_id", "gold"]].rename(columns={"gold": "my_label"}), on="part_id", how="left")
    seg["in_420"] = seg["my_label"].notna()
    when = pd.to_datetime(seg["review_date"], errors="coerce")
    inside = (when >= pd.Timestamp(args.start)) & (when <= pd.Timestamp(args.end))
    if args.source.lower() != "all":
        inside &= seg["source"].fillna("").str.contains(args.source, case=False, regex=False)      # "Google Play" matches "Google Play Store"
        if int(inside.sum()) == 0:
            sys.exit(f"No parts match source '{args.source}'. The sources in your data are: {sorted(seg['source'].dropna().unique())}")
    print(f"\nDashboard window {args.start} to {args.end}, source: {args.source}. Keeping {int(inside.sum())} of {len(seg)} parts "
          f"({int((~inside).sum())} are outside the window, from another source, or have no date).")
    perf["window"] = {"from": args.start, "to": args.end, "source": args.source, "parts": int(inside.sum())}
    (OUT / "performance.json").write_text(json.dumps(perf, indent=2))
    seg = seg[inside]
    keep = ["part_id", "review_id", "telco", "source", "month", "aspect", "segment_text", "dict_en", "in_420", "my_label",
            "pred_manglish", "pred_english", "pred_malay"]
    seg[keep].rename(columns={"segment_text": "text", "dict_en": "text_en"}).to_csv(OUT / "parts_dashboard.csv", index=False)
    print(f"\nSaved in {OUT}:")
    for f in sorted(OUT.iterdir()):
        print(f"   {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print("\nNow run:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
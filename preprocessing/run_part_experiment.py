"""
run_part_experiment.py - the experiment on the corrected parts: ONE SVM PER TEXT VERSION, each trained and tested on its own text.

Versions (spellchecker OFF, full translation left out):
  Manglish : raw | cleaned | normalized
  English  : normalized + dictionary (all words) | normalized + dictionary (telco terms only)
  Malay    : normalized + dictionary (all words) | normalized + dictionary (telco terms only)

How it is done
  1. Labels: XLM-R's label of the RAW part (label_parts_xlmr.py). Every version of a part shares that label.
  2. Split 80 / 20 BY REVIEW (all parts of one review go to the same side), the same in every month and telco.
  3. Training set: balanced (the same number of negative / neutral / positive, as many as the rarest class allows).
     Test set: natural mix (not balanced), so the score is realistic.
  4. Parts whose cleaned text is in both sets are removed from training.
  5. For every version: TF-IDF (1-2 words) + linear SVM, trained on the training parts in that version,
     tested on the test parts in the SAME version.

  python3 preprocessing/run_part_experiment.py
  python3 preprocessing/run_part_experiment.py --train-cap 3000     smaller / faster (parts per class)

Reads : review_segments_raw.csv, part_xlmr_labels.csv, 4_dictionary_mapped_reviews.csv (telco, review_date, source)
        segment_check_sheet.csv (optional: your 300 hand-checked segments, scored too)
Writes: part_split.csv  part_versions.csv  part_results.csv  part_results.xlsx  (all in data/processed)
        sheets in the xlsx: SVM per version | Per class | Confusion matrices | Notes
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manglish_normalizer as nz                                      # noqa: E402
import text_versions as tv                                            # noqa: E402
from dictionary_mapper import load_dictionary, map_text               # noqa: E402

DATA = tv.DATA
SEED = 42
VERSIONS = [("Manglish", "raw", "Raw"), ("Manglish", "cleaned", "Cleaned"), ("Manglish", "norm", "Normalized"),
            ("English", "dict_en", "Dictionary (all words)"), ("English", "dict_en_telco", "Dictionary (telco terms only)"),
            ("Malay", "dict_ms", "Dictionary (all words)"), ("Malay", "dict_ms_telco", "Dictionary (telco terms only)")]
CLASSES = ["negative", "neutral", "positive"]


def make_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LinearSVC(C=1.0, random_state=SEED))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cap", type=int, default=0, help="at most this many training parts per class (0 = as many as possible)")
    ap.add_argument("--test-share", type=float, default=0.2)
    args = ap.parse_args()

    seg = pd.read_csv(DATA / "review_segments_raw.csv", dtype={"review_id": str})
    seg["segment_text"] = seg["segment_text"].fillna("").astype(str)
    seg["part_id"] = seg["review_id"] + "_" + seg["segment_id"].astype(str)
    lab = pd.read_csv(DATA / "part_xlmr_labels.csv", dtype={"part_id": str})
    meta = pd.read_csv(DATA / "4_dictionary_mapped_reviews.csv", usecols=["review_id", "telco", "review_date", "source"],
                       dtype={"review_id": str})
    df = seg.merge(lab, on="part_id", how="inner").merge(meta, on="review_id", how="left")
    df = df[df["segment_text"].str.strip() != ""].copy()
    print(f"{len(df)} labelled parts from {df['review_id'].nunique()} reviews "
          f"(of {len(seg)} parts; {len(seg) - len(df)} not labelled yet or empty)")
    print("XLM-R labels, all parts:", df["xlmr_label"].value_counts().to_dict())
    df["month"] = pd.to_datetime(df["review_date"], errors="coerce").dt.to_period("M").astype(str)

    # ---- 2. split by review, spread over telco and month ---------------------------------------------------------
    rev = df.drop_duplicates("review_id")[["review_id", "telco", "month"]]
    test_ids = set(rev.groupby(["telco", "month"], group_keys=False).sample(frac=args.test_share, random_state=SEED)["review_id"])
    df["split"] = np.where(df["review_id"].isin(test_ids), "test", "train")
    df["cleaned"] = [tv.clean_text(t) for t in df["segment_text"]]
    test_texts = set(df.loc[df["split"] == "test", "cleaned"])
    leak = (df["split"] == "train") & df["cleaned"].isin(test_texts)
    df = df[~leak]
    print(f"Removed {int(leak.sum())} training parts whose text is also in the test set.")

    train_all = df[df["split"] == "train"]
    k = int(train_all["xlmr_label"].value_counts().reindex(CLASSES).fillna(0).min())
    if args.train_cap:
        k = min(k, args.train_cap)
    train = (train_all.groupby("xlmr_label", group_keys=False).sample(n=k, random_state=SEED))
    test = df[df["split"] == "test"]
    print(f"\nTraining set: {k} per class = {len(train)} parts (balanced).   Test set: {len(test)} parts (natural mix)")
    print("  train classes:", train["xlmr_label"].value_counts().to_dict())
    print("  test classes :", test["xlmr_label"].value_counts().to_dict())
    base = test["xlmr_label"].value_counts().max() / len(test) * 100
    print(f"  always-the-biggest-class baseline on the test set: {base:.1f}%")
    print("  parts per year (train / test):", train["month"].str[:4].value_counts().sort_index().to_dict(),
          test["month"].str[:4].value_counts().sort_index().to_dict())
    print("  parts per telco (train / test):", train["telco"].value_counts().to_dict(), test["telco"].value_counts().to_dict())
    used = pd.concat([train, test]).reset_index(drop=True)
    used[["part_id", "review_id", "split", "telco", "month", "aspect", "xlmr_label"]].to_csv(DATA / "part_split.csv", index=False)

    # ---- 3. make the text versions --------------------------------------------------------------------------------
    print("\nMaking the text versions ...")
    counts = tv.word_counts()
    used["raw"] = used["segment_text"]
    used["norm"] = tv.normalize_texts(used["cleaned"], False, counts)
    full = load_dictionary()
    telco, n_telco = tv.telco_only_lookup()
    for lang, code in (("en", "english"), ("ms", "malay")):
        used[f"dict_{lang}"] = [map_text(t, full, code) for t in used["norm"]]
        used[f"dict_{lang}_telco"] = [map_text(t, telco, code) for t in used["norm"]]
    nz.USE_SPELLCHECKER = False
    used.drop(columns=["segment_text"]).to_csv(DATA / "part_versions.csv", index=False, encoding="utf-8-sig")

    # ---- 4. one SVM per version ----------------------------------------------------------------------------------
    tr, te = used[used["split"] == "train"], used[used["split"] == "test"]
    rows, raw_ok, models, perclass, confs = [], None, {}, [], []
    for group, col, name in VERSIONS:
        model = make_model().fit(tr[col], tr["xlmr_label"])
        models[col] = model
        pred = model.predict(te[col])
        pr, rc, f1c, sup = precision_recall_fscore_support(te["xlmr_label"], pred, labels=CLASSES, zero_division=0)
        for c, a, b, d, n in zip(CLASSES, pr, rc, f1c, sup):
            perclass.append({"Text": group, "Version": name, "Class": c, "Precision %": round(a * 100, 1),
                             "Recall %": round(b * 100, 1), "F1 %": round(d * 100, 1), "Test parts": int(n)})
        cm = confusion_matrix(te["xlmr_label"], pred, labels=CLASSES)
        confs.append((f"{group} - {name}", cm))
        ok = (pred == te["xlmr_label"].values)
        row = {"Text": group, "Version": name,
               "Accuracy %": round(accuracy_score(te["xlmr_label"], pred) * 100, 1),
               "Macro-F1 %": round(f1_score(te["xlmr_label"], pred, average="macro") * 100, 1)}
        for c, f in zip(CLASSES, f1_score(te["xlmr_label"], pred, average=None, labels=CLASSES)):
            row[f"F1 {c} %"] = round(f * 100, 1)
        if col == "raw":
            raw_ok = ok
        fixed, broken = int((ok & ~raw_ok).sum()), int((~ok & raw_ok).sum())
        row.update({"Fixed vs raw": fixed, "Broken vs raw": broken,
                    "p-value": round(binomtest(fixed, fixed + broken, 0.5).pvalue, 3) if fixed + broken else 1.0})
        rows.append(row)
        print(f"  {group:9s} {name:32s} accuracy {row['Accuracy %']:5.1f}%   macro-F1 {row['Macro-F1 %']:5.1f}%")
    # ---- 5. your 300 hand-checked segments (never used for training; texts that are also in training are left out) ----
    sheet_path = DATA / "segment_check_sheet.csv"
    manual_note = "segment_check_sheet.csv not found: the scoring against your own labels was skipped."
    if sheet_path.exists():
        lw = {"positive": "positive", "pos": "positive", "p": "positive", "+": "positive", "negative": "negative",
              "neg": "negative", "n": "negative", "-": "negative", "neutral": "neutral", "neu": "neutral", "0": "neutral"}
        sh = pd.read_csv(sheet_path, encoding="utf-8-sig", dtype=str).fillna("")
        sh["gold"] = sh["manual_label"].str.strip().str.lower().map(lw)
        sh = sh.dropna(subset=["gold"]).reset_index(drop=True)
        sh["cleaned"] = [tv.clean_text(t) for t in sh["segment_text"]]
        seen = set(tr["cleaned"])
        n_all = len(sh)
        sh = sh[~sh["cleaned"].isin(seen)].reset_index(drop=True)
        sh["raw"] = sh["segment_text"]
        sh["norm"] = tv.normalize_texts(sh["cleaned"], False, counts)
        for lang, code in (("en", "english"), ("ms", "malay")):
            sh[f"dict_{lang}"] = [map_text(t, full, code) for t in sh["norm"]]
            sh[f"dict_{lang}_telco"] = [map_text(t, telco, code) for t in sh["norm"]]
        nz.USE_SPELLCHECKER = False
        print(f"\nYour hand-checked segments: {len(sh)} of {n_all} used ({n_all - len(sh)} left out: same text is in the training set)")
        xl_acc = None
        if "xlmr_label" in sh:
            xl_acc = round(accuracy_score(sh["gold"], sh["xlmr_label"]) * 100, 1)
            print(f"  XLM-R itself against your labels on these segments: {xl_acc}%  (it labelled them first, you then checked)")
        for r, (group, col, name) in zip(rows, VERSIONS):
            pred = models[col].predict(sh[col])
            r["Accuracy vs YOUR labels %"] = round(accuracy_score(sh["gold"], pred) * 100, 1)
            r["Macro-F1 vs YOUR labels %"] = round(f1_score(sh["gold"], pred, average="macro", labels=CLASSES, zero_division=0) * 100, 1)
            print(f"  {group:9s} {name:32s} against your labels: accuracy {r['Accuracy vs YOUR labels %']:5.1f}%   "
                  f"macro-F1 {r['Macro-F1 vs YOUR labels %']:5.1f}%")
        base_g = sh["gold"].value_counts().max() / len(sh) * 100
        manual_note = (f"Against YOUR labels: {len(sh)} hand-checked segments (never used for training; {n_all - len(sh)} left out because the same text "
                       f"is in the training set). Always-negative would score {base_g:.1f}% on them. XLM-R itself: {xl_acc}%.")
    res = pd.DataFrame(rows)
    res.to_csv(DATA / "part_results.csv", index=False)
    print("\nConfusion matrices (rows = XLM-R label, columns = SVM prediction):")
    for title, cm in confs:
        print(f"  {title}")
        for c, row in zip(CLASSES, cm):
            print(f"      {c:9s} -> " + "  ".join(f"{k[:3]} {v:6d}" for k, v in zip(CLASSES, row)))
    pc = pd.DataFrame(perclass)
    pc.to_csv(DATA / "part_results_per_class.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "part_results.xlsx") as xl:
            res.to_excel(xl, sheet_name="SVM per version", index=False)
            pc.to_excel(xl, sheet_name="Per class", index=False)
            pd.concat([pd.DataFrame([{"Version": t, "Row = XLM-R label": c, **{f"SVM says {k}": int(v) for k, v in zip(CLASSES, row)}}
                                     for c, row in zip(CLASSES, cm)]) for t, cm in confs]).to_excel(
                xl, sheet_name="Confusion matrices", index=False)
            pd.DataFrame({"Note": [
                f"Each version: its own SVM, trained on {len(tr)} balanced training parts and tested on {len(te)} test parts, in the SAME text version.",
                "Labels = XLM-R's label of the raw part. The score means 'agreement with XLM-R', not blind accuracy.",
                "Split by review (80/20), spread over telco and month. Test set is NOT balanced.",
                f"Always-the-biggest-class baseline on this test set: {base:.1f}%.",
                "p-value: exact McNemar test vs the raw version. Below 0.05 = unlikely to be luck.", manual_note]}).to_excel(xl, sheet_name="Notes", index=False)
    except Exception as e:                                   # openpyxl missing: the csv is enough
        print("Could not write the xlsx:", e)
    print(f"\nSaved part_results.csv / part_results.xlsx / part_split.csv / part_versions.csv in {DATA}")


if __name__ == "__main__":
    main()
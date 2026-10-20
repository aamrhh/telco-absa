"""
run_full_tables.py  --  PAGE 2: the SVM trained and tested on the FULL data, labelled by each auto-labeler.
(Your 420 labels are NOT used here. They are used on page 1 and page 3 only.)

For every text version, TWO SVMs:
   - trained on XLM-R's labels, and tested on XLM-R's labels (held-out test reviews)
   - trained on DistilBERT's labels, and tested on DistilBERT's labels (the same held-out test reviews)
The test set is the natural mix (not balanced), like real data. The score means: "how well the SVM copies its labeler".

Run (after label_parts_distilbert.py has finished):
    python3 preprocessing/run_full_tables.py

Needs : review_segments_raw.csv, part_xlmr_labels.csv, part_distil_labels.csv, 4_dictionary_mapped_reviews.csv
Writes: data/processed/full_tables.xlsx  (sheets: Malay, English, Manglish, Notes)  and  full_tables.csv

Rules (same as before):
  1. Split the reviews 80/20 by review (same seed, so the same test reviews as before).
  2. Remove training parts whose text is also in the test part.
  3. Training is balanced per labeler: the same number of negative, neutral and positive parts by THAT labeler's labels.
  4. One SVM per labeler per text version.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manglish_normalizer as nz                                      # noqa: E402
import text_versions as tv                                            # noqa: E402
from dictionary_mapper import load_dictionary, map_text               # noqa: E402

DATA = tv.DATA
SEED = 42
CLASSES = ["negative", "neutral", "positive"]
LABELERS = [("xlmr_label", "XLM-R"), ("distil_label", "DistilBERT")]
TABLES = {
    "Malay": [("Raw", "raw"), ("Cleaned only", "cleaned"), ("Normalized (after cleaning)", "norm"), ("Malay dictionary", "dict_ms")],
    "English": [("Raw", "raw"), ("Cleaned only", "cleaned"), ("Normalized (after cleaning)", "norm"), ("English dictionary", "dict_en")],
    "Manglish": [("Raw", "raw"), ("Cleaned only", "cleaned")],
}


def make_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LinearSVC(C=1.0, random_state=SEED))


def main():
    # ---- 1. the parts, the labels, the same split as before ---------------------------------------------------------
    seg = pd.read_csv(DATA / "review_segments_raw.csv", dtype={"review_id": str})
    seg["segment_text"] = seg["segment_text"].fillna("").astype(str)
    seg["part_id"] = seg["review_id"] + "_" + seg["segment_id"].astype(str)
    xl = pd.read_csv(DATA / "part_xlmr_labels.csv", dtype={"part_id": str})[["part_id", "xlmr_label"]]
    di = pd.read_csv(DATA / "part_distil_labels.csv", dtype={"part_id": str})[["part_id", "distil_label"]]
    meta = pd.read_csv(DATA / "4_dictionary_mapped_reviews.csv", usecols=["review_id", "telco", "review_date", "source"],
                       dtype={"review_id": str})
    df = seg.merge(xl, on="part_id", how="inner").merge(meta, on="review_id", how="left")
    df = df[df["segment_text"].str.strip() != ""].copy()
    df["month"] = pd.to_datetime(df["review_date"], errors="coerce").dt.to_period("M").astype(str)
    rev = df.drop_duplicates("review_id")[["review_id", "telco", "month"]]
    test_ids = set(rev.groupby(["telco", "month"], group_keys=False).sample(frac=0.2, random_state=SEED)["review_id"])
    df["split"] = np.where(df["review_id"].isin(test_ids), "test", "train")
    df["cleaned"] = [tv.clean_text(t) for t in df["segment_text"]]
    test_texts = set(df.loc[df["split"] == "test", "cleaned"])
    df = df[~((df["split"] == "train") & df["cleaned"].isin(test_texts))]
    df = df.merge(di, on="part_id", how="left")
    print(f"{len(df)} parts.  DistilBERT labelled {df['distil_label'].notna().sum()} of them.")

    # ---- 2. the held-out test parts (natural mix) -------------------------------------------------------------------
    test = df[df["split"] == "test"]
    print(f"\nTest parts (held-out reviews): {len(test)}")
    for col, name in LABELERS:
        t = test[col].dropna()
        print(f"  {name:10s} test labels: {t.value_counts().to_dict()}   always-biggest-class: {t.value_counts().max() / len(t) * 100:.1f}%")

    # ---- 3. balanced training sets, one per labeler -----------------------------------------------------------------
    train_all = df[df["split"] == "train"]
    trains = {}
    for col, name in LABELERS:
        t = train_all[train_all[col].isin(CLASSES)]
        k = int(t[col].value_counts().reindex(CLASSES).fillna(0).min())
        trains[col] = t.groupby(col, group_keys=False).sample(n=k, random_state=SEED)
        print(f"  {name:10s} training: {k} per class = {len(trains[col])} parts")
    need = pd.concat([trains[c] for c, _ in LABELERS] + [test]).drop_duplicates("part_id").reset_index(drop=True)

    # ---- 4. text versions (only for the parts we need) ---------------------------------------------------------------
    print(f"\nMaking the text versions for {len(need)} parts ...")
    counts = tv.word_counts()
    full = load_dictionary()
    need["raw"] = need["segment_text"]
    need["norm"] = tv.normalize_texts(need["cleaned"], False, counts)
    need["dict_en"] = [map_text(t, full, "english") for t in need["norm"]]
    need["dict_ms"] = [map_text(t, full, "malay") for t in need["norm"]]
    nz.USE_SPELLCHECKER = False
    vers = need.set_index("part_id")
    te = vers[vers["split"] == "test"]

    # ---- 5. train every SVM, test it on the held-out parts, with the SAME labeler's labels --------------------------
    results = {}
    for vcol in ("raw", "cleaned", "norm", "dict_en", "dict_ms"):
        for col, name in LABELERS:
            tr = vers.loc[trains[col]["part_id"]]
            tt = te[te[col].isin(CLASSES)]
            m = make_model().fit(tr[vcol].fillna(""), tr[col])
            pred = m.predict(tt[vcol].fillna(""))
            results[(name, vcol)] = (accuracy_score(tt[col], pred),
                                     f1_score(tt[col], pred, average="macro", labels=CLASSES, zero_division=0),
                                     f1_score(tt[col], pred, average=None, labels=CLASSES, zero_division=0), len(tt))
            print(f"  SVM on {name:10s} labels, version {vcol:8s}: right {results[(name, vcol)][0] * 100:5.1f}%   "
                  f"fair score {results[(name, vcol)][1] * 100:5.1f}%")
    for col, name in LABELERS:                      # "always the biggest class" on each labeler's own test labels
        t = te[col].dropna()
        results[(name, "baseline")] = (t.value_counts().max() / len(t), None, None, len(t))

    # ---- 6. the three tables ---------------------------------------------------------------------------------------
    long_rows, sheets = [], {}
    for table, versions in TABLES.items():
        out = []
        row = {"Text version": "Always the biggest class (baseline)"}
        for _, name in LABELERS:
            a, _, _, n = results[(name, "baseline")]
            row[f"{name}: % right"], row[f"{name}: fair score"] = round(a * 100, 1), "-"
            long_rows.append({"Table": table, "Row": "Baseline", "Labeler": name, "Accuracy %": round(a * 100, 1), "Parts": n})
        out.append(row)
        for label, vcol in versions:
            row = {"Text version": label}
            for _, name in LABELERS:
                a, f, pf, n = results[(name, vcol)]
                row[f"{name}: % right"], row[f"{name}: fair score"] = round(a * 100, 1), round(f * 100, 1)
                long_rows.append({"Table": table, "Row": label, "Labeler": name, "Accuracy %": round(a * 100, 1),
                                  "Macro-F1 %": round(f * 100, 1), "F1 negative %": round(pf[0] * 100, 1),
                                  "F1 neutral %": round(pf[1] * 100, 1), "F1 positive %": round(pf[2] * 100, 1), "Parts": n})
            out.append(row)
        sheets[table] = pd.DataFrame(out)
        print(f"\n=== {table} table (SVM trained and tested on the full data, same labeler's labels) ===")
        print(sheets[table].to_string(index=False))
    pd.DataFrame(long_rows).to_csv(DATA / "full_tables.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "full_tables.xlsx") as xlw:
            for table, d in sheets.items():
                d.to_excel(xlw, sheet_name=table, index=False)
            pd.DataFrame({"How to read": [
                "% right = out of all held-out test parts, how many the SVM gave the same label as its own labeler (XLM-R or DistilBERT).",
                "fair score = macro-F1: the average of the score for negative, neutral and positive.",
                "The test set is the natural mix (mostly negative), so compare with the baseline row.",
                "This page says how well the SVM copies its labeler. It does not say the labeler is right: page 1 tests the labelers against my labels.",
                "Training is balanced per labeler (the same number of negative, neutral and positive parts)."]}).to_excel(
                xlw, sheet_name="Notes", index=False)
    except Exception as e:                                                       # noqa: BLE001
        print("Could not write the xlsx:", e)
    print(f"\nSaved full_tables.xlsx and full_tables.csv in {DATA}")


if __name__ == "__main__":
    main()
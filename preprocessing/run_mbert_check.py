"""
run_mbert_check.py  --  try multilingual BERT (mBERT) as another auto-labeler, tested on your 420 blind labels (same as page 1).

Model: nlptown/bert-base-multilingual-uncased-sentiment   (mBERT fine-tuned on product reviews, MIT licence)
Plain mBERT has no sentiment ability by itself, so we use this fine-tuned version.
It gives a STAR RATING from 1 to 5. Trained on English, Dutch, German, French, Spanish and Italian reviews. NOT Malay.
We turn stars into classes with a rule fixed BEFORE looking at your labels:
    1 or 2 stars = negative      3 stars = neutral      4 or 5 stars = positive
Do not change this rule after seeing the result.

Run:     python3 preprocessing/run_mbert_check.py
Needs:   review_segments_raw.csv, blind_label_sheet.csv (filled in), internet the first time (downloads about 700 MB)
Writes:  data/processed/mbert_check.xlsx  and  mbert_check.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manglish_normalizer as nz                                      # noqa: E402
import text_versions as tv                                            # noqa: E402
from dictionary_mapper import load_dictionary, map_text               # noqa: E402

DATA = tv.DATA
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
CLASSES = ["negative", "neutral", "positive"]
LABELS = {"negative": "negative", "neg": "negative", "n": "negative", "-": "negative",
          "neutral": "neutral", "neu": "neutral", "0": "neutral",
          "positive": "positive", "pos": "positive", "p": "positive", "+": "positive"}
VERSIONS = [("Raw", "raw"), ("Cleaned only", "cleaned"), ("Normalized (after cleaning)", "norm"),
            ("English dictionary", "dict_en"), ("Malay dictionary", "dict_ms")]
STAR_MAP = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def label_texts(texts):
    from transformers import pipeline
    clf = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID, truncation=True, max_length=256)
    res = clf([t if str(t).strip() else "." for t in texts], batch_size=16)
    stars = [int(r["label"].split()[0]) for r in res]
    return np.array([STAR_MAP[s] for s in stars]), np.array(stars)


def main():
    seg = pd.read_csv(DATA / "review_segments_raw.csv", dtype={"review_id": str})
    seg["part_id"] = seg["review_id"] + "_" + seg["segment_id"].astype(str)
    seg["segment_text"] = seg["segment_text"].fillna("").astype(str)
    sh = pd.read_csv(DATA / "blind_label_sheet.csv", encoding="utf-8-sig", dtype=str).fillna("")
    sh["gold"] = sh["my_label"].str.strip().str.lower().map(LABELS)
    sh = sh.dropna(subset=["gold"])
    d = seg.merge(sh[["part_id", "gold"]], on="part_id").reset_index(drop=True)
    print(f"{len(d)} of your {len(sh)} labelled parts found.  Your labels: {d['gold'].value_counts().to_dict()}")

    print("Making the text versions ...")
    counts = tv.word_counts()
    full = load_dictionary()
    d["raw"] = d["segment_text"]
    d["cleaned"] = [tv.clean_text(t) for t in d["raw"]]
    d["norm"] = tv.normalize_texts(d["cleaned"], False, counts)
    d["dict_en"] = [map_text(t, full, "english") for t in d["norm"]]
    d["dict_ms"] = [map_text(t, full, "malay") for t in d["norm"]]
    nz.USE_SPELLCHECKER = False

    rows = []
    for label, col in VERSIONS:
        print(f"Labelling {label} ...")
        pred, stars = label_texts(d[col].tolist())
        acc = accuracy_score(d["gold"], pred)
        mf = f1_score(d["gold"], pred, average="macro", labels=CLASSES, zero_division=0)
        f = f1_score(d["gold"], pred, average=None, labels=CLASSES, zero_division=0)
        rows.append({"Text version": label, "% right": round(acc * 100, 1), "Fair score": round(mf * 100, 1),
                     "F1 negative": round(f[0] * 100, 1), "F1 neutral": round(f[1] * 100, 1), "F1 positive": round(f[2] * 100, 1)})
        print(f"   right {acc * 100:5.1f}%  fair {mf * 100:5.1f}%   says {dict(zip(*np.unique(pred, return_counts=True)))}   stars {dict(zip(*np.unique(stars, return_counts=True)))}")

    tab = pd.DataFrame(rows)
    print(f"\n=== mBERT (star rating) against your {len(d)} blind labels ===")
    print(tab.to_string(index=False))
    print("\nCompare with page 1 (XLM-R, DistilBERT) in labeler_check.xlsx and mDeBERTa in deberta_check.xlsx.")
    tab.to_csv(DATA / "mbert_check.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "mbert_check.xlsx") as xl:
            tab.to_excel(xl, sheet_name="mBERT", index=False)
            pd.DataFrame({"How to read": [
                f"Model: {MODEL_ID}. It gives 1 to 5 stars.",
                "Rule fixed before looking: 1-2 stars negative, 3 stars neutral, 4-5 stars positive.",
                "Trained on product reviews in 6 European languages, not Malay: low scores are expected.",
                "% right = same as my label. Fair score = macro-F1."]}).to_excel(xl, sheet_name="Notes", index=False)
    except Exception as e:                                                       # noqa: BLE001
        print("Could not write the xlsx:", e)
    print(f"\nSaved mbert_check.xlsx and mbert_check.csv in {DATA}")


if __name__ == "__main__":
    main()
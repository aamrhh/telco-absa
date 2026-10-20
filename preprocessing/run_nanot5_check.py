"""
run_nanot5_check.py  --  Mesolitica's NanoT5 (the labeller used by Tan et al. 2025), tested on your 420 blind labels (same as page 1).

It is the model inside Malaya: malaya.sentiment.huggingface()  ->  mesolitica/sentiment-analysis-nanot5-small-malaysian-cased
Trained on Malaysian text (standard Malay and social media), gives negative / neutral / positive directly. No cut-off or star rule needed.

Run:     python3 preprocessing/run_nanot5_check.py
Needs:   review_segments_raw.csv, blind_label_sheet.csv (filled in), Malaya installed (you already have it)
Writes:  data/processed/nanot5_check.xlsx  and  nanot5_check.csv
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

if not hasattr(inspect, "getargspec"):                 # Malaya needs this on Python 3.11
    inspect.getargspec = inspect.getfullargspec

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
_MODEL = None


def label_texts(texts):
    global _MODEL
    if _MODEL is None:
        import malaya
        _MODEL = malaya.sentiment.huggingface()
    texts = [t if str(t).strip() else "." for t in texts]
    out = []
    for i in range(0, len(texts), 32):
        for r in _MODEL.predict(texts[i:i + 32]):
            r = str(r.get("label") if isinstance(r, dict) else r).lower()
            out.append("positive" if r.startswith("pos") else "negative" if r.startswith("neg") else "neutral")
    return np.array(out)


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
        pred = label_texts(d[col].tolist())
        acc = accuracy_score(d["gold"], pred)
        mf = f1_score(d["gold"], pred, average="macro", labels=CLASSES, zero_division=0)
        f = f1_score(d["gold"], pred, average=None, labels=CLASSES, zero_division=0)
        rows.append({"Text version": label, "% right": round(acc * 100, 1), "Fair score": round(mf * 100, 1),
                     "F1 negative": round(f[0] * 100, 1), "F1 neutral": round(f[1] * 100, 1), "F1 positive": round(f[2] * 100, 1)})
        print(f"   right {acc * 100:5.1f}%  fair {mf * 100:5.1f}%   says {dict(zip(*np.unique(pred, return_counts=True)))}")

    tab = pd.DataFrame(rows)
    print(f"\n=== NanoT5 (Malaya) against your {len(d)} blind labels ===")
    print(tab.to_string(index=False))
    print("\nCompare with page 1 (XLM-R, DistilBERT) in labeler_check.xlsx, mDeBERTa in deberta_check.xlsx and mBERT in mbert_check.xlsx.")
    tab.to_csv(DATA / "nanot5_check.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "nanot5_check.xlsx") as xl:
            tab.to_excel(xl, sheet_name="NanoT5", index=False)
            pd.DataFrame({"How to read": [
                "Model: Mesolitica NanoT5 small, through malaya.sentiment.huggingface() (the same labeller family as Tan et al. 2025).",
                "It gives negative / neutral / positive directly. Trained on Malaysian text.",
                "% right = same as my label. Fair score = macro-F1."]}).to_excel(xl, sheet_name="Notes", index=False)
    except Exception as e:                                                       # noqa: BLE001
        print("Could not write the xlsx:", e)
    print(f"\nSaved nanot5_check.xlsx and nanot5_check.csv in {DATA}")


if __name__ == "__main__":
    main()
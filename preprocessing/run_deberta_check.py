"""
run_deberta_check.py  --  try mDeBERTa as a THIRD auto-labeler, tested on your 420 blind labels (same as page 1).

Model: agentlans/mdeberta-v3-base-sentiment   (multilingual DeBERTa, MIT licence)
IMPORTANT: this model does NOT give "negative / neutral / positive". It gives ONE number (a sentiment score):
    below 0 = negative side, above 0 = positive side, near 0 = neutral.
So we must pick a "neutral zone": score between -CUT and +CUT = neutral.
It was trained on English sentences (translated Tatoeba), not on reviews and not on Malay. It may do badly. That is fine to report.

Fairness rule: the main CUT is chosen BEFORE looking at your labels (default 0.5). The other cuts are shown only to see
how much the result moves. Do not pick the best cut on your 420 labels and call it the result.

Run:     python3 preprocessing/run_deberta_check.py
         python3 preprocessing/run_deberta_check.py --cut 0.5
Needs:   review_segments_raw.csv, blind_label_sheet.csv (filled in), internet the first time (downloads about 1 GB)
Writes:  data/processed/deberta_check.xlsx  and  deberta_check.csv
"""
import argparse
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
MODEL_ID = "agentlans/mdeberta-v3-base-sentiment"
CLASSES = ["negative", "neutral", "positive"]
LABELS = {"negative": "negative", "neg": "negative", "n": "negative", "-": "negative",
          "neutral": "neutral", "neu": "neutral", "0": "neutral",
          "positive": "positive", "pos": "positive", "p": "positive", "+": "positive"}
VERSIONS = [("Raw", "raw"), ("Cleaned only", "cleaned"), ("Normalized (after cleaning)", "norm"),
            ("English dictionary", "dict_en"), ("Malay dictionary", "dict_ms")]
EXTRA_CUTS = [0.25, 1.0]


def score_texts(texts):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev).eval()
    out = []
    texts = [t if str(t).strip() else "." for t in texts]
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", truncation=True, padding=True, max_length=256).to(dev)
        with torch.no_grad():
            lg = model(**enc).logits.float().cpu().numpy()
        if lg.shape[-1] != 1:
            sys.exit(f"The model gave {lg.shape[-1]} numbers per text, I expected 1. Tell Claude.")
        out.extend(lg[:, 0].tolist())
    return np.array(out)


def to_class(score, cut):
    return np.where(score < -cut, "negative", np.where(score > cut, "positive", "neutral"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=float, default=0.5, help="neutral zone is -cut .. +cut (default 0.5, fixed before looking)")
    args = ap.parse_args()

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

    rows, scores = [], {}
    for label, col in VERSIONS:
        print(f"Scoring {label} ...")
        scores[col] = score_texts(d[col].tolist())
        s = scores[col]
        print(f"   score range {s.min():.2f} to {s.max():.2f}, middle 50% between {np.percentile(s, 25):.2f} and {np.percentile(s, 75):.2f}")
        for cut in [args.cut] + EXTRA_CUTS:
            pred = to_class(s, cut)
            acc = accuracy_score(d["gold"], pred)
            mf = f1_score(d["gold"], pred, average="macro", labels=CLASSES, zero_division=0)
            f = f1_score(d["gold"], pred, average=None, labels=CLASSES, zero_division=0)
            rows.append({"Text version": label, "Neutral zone (+/-)": cut, "Main": cut == args.cut,
                         "% right": round(acc * 100, 1), "Fair score": round(mf * 100, 1),
                         "F1 negative": round(f[0] * 100, 1), "F1 neutral": round(f[1] * 100, 1), "F1 positive": round(f[2] * 100, 1)})
            print(f"   zone +/-{cut:<5}: right {acc * 100:5.1f}%  fair {mf * 100:5.1f}%   says {dict(zip(*np.unique(pred, return_counts=True)))}")

    res = pd.DataFrame(rows)
    main_tab = res[res["Main"]][["Text version", "% right", "Fair score", "F1 negative", "F1 neutral", "F1 positive"]]
    print(f"\n=== mDeBERTa against your {len(d)} blind labels (neutral zone +/-{args.cut}, chosen before looking) ===")
    print(main_tab.to_string(index=False))
    print("\nCompare with page 1 (XLM-R and DistilBERT) in labeler_check.xlsx.")
    res.to_csv(DATA / "deberta_check.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "deberta_check.xlsx") as xl:
            main_tab.to_excel(xl, sheet_name="mDeBERTa", index=False)
            res.to_excel(xl, sheet_name="All cuts", index=False)
            pd.DataFrame({"How to read": [
                f"Model: {MODEL_ID}. It gives one score, not three classes.",
                f"Neutral zone: score between -{args.cut} and +{args.cut} = neutral. Chosen before looking at the labels.",
                "Other zones are in sheet 'All cuts' only to show how much the result moves.",
                "Trained on English sentences, not on reviews or Malay: low scores are expected.",
                "% right = same as my label. Fair score = macro-F1."]}).to_excel(xl, sheet_name="Notes", index=False)
    except Exception as e:                                                       # noqa: BLE001
        print("Could not write the xlsx:", e)
    print(f"\nSaved deberta_check.xlsx and deberta_check.csv in {DATA}")


if __name__ == "__main__":
    main()
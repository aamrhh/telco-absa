"""
run_labeler_check.py  --  how good are the two AUTO-LABELERS (XLM-R and DistilBERT) against YOUR 420 blind labels?

This is the "first page": it tests the labelers themselves, not the SVM.
Each labeler is run on every text version of your 420 parts:
    Raw, Cleaned only, Normalized, English dictionary, Malay dictionary
so you can see whether the cleaning / normalizer / dictionary help the labeler too.

Run:     python3 preprocessing/run_labeler_check.py
Needs:   review_segments_raw.csv, blind_label_sheet.csv (filled in)
         (it downloads XLM-R and DistilBERT if they are not on your Mac yet; only 420 parts x 5 versions, so a few minutes)
Writes:  data/processed/labeler_check.xlsx   (sheet "Labelers" = the table, sheet "Per class" = more detail)
         data/processed/labeler_check.csv
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

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
MODELS = [("XLM-R", "cardiffnlp/twitter-xlm-roberta-base-sentiment", {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}),
          ("DistilBERT", "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
           {"negative": "negative", "neutral": "neutral", "positive": "positive"})]
VERSIONS = [("Raw", "raw"), ("Cleaned only", "cleaned"), ("Normalized (after cleaning)", "norm"),
            ("English dictionary", "dict_en"), ("Malay dictionary", "dict_ms")]


def label_texts(model_id, fix, texts):
    from transformers import pipeline
    clf = pipeline("text-classification", model=model_id, tokenizer=model_id, truncation=True, max_length=256)
    res = clf([t if str(t).strip() else "." for t in texts], batch_size=16)
    return [fix.get(r["label"].lower(), r["label"].lower()) for r in res]


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

    rows, pc_rows, table = [], [], {label: {"Text version": label} for label, _ in VERSIONS}
    for name, model_id, fix in MODELS:
        for label, col in VERSIONS:
            pred = label_texts(model_id, fix, d[col].tolist())
            acc = accuracy_score(d["gold"], pred)
            mf = f1_score(d["gold"], pred, average="macro", labels=CLASSES, zero_division=0)
            f = f1_score(d["gold"], pred, average=None, labels=CLASSES, zero_division=0)
            table[label][f"{name}: % right"] = round(acc * 100, 1)
            table[label][f"{name}: fair score"] = round(mf * 100, 1)
            rows.append({"Labeler": name, "Text version": label, "Accuracy %": round(acc * 100, 1), "Macro-F1 %": round(mf * 100, 1),
                         "F1 negative %": round(f[0] * 100, 1), "F1 neutral %": round(f[1] * 100, 1), "F1 positive %": round(f[2] * 100, 1),
                         "Parts": len(d)})
            for c in CLASSES:                      # when the labeler says c, how often do you agree?
                sel = [p == c for p in pred]
                n = sum(sel)
                k = int(((d["gold"] == c) & pd.Series(sel)).sum())
                pc_rows.append({"Labeler": name, "Text version": label, "Labeler says": c, "Times": n,
                                "You agree %": round(k / n * 100, 1) if n else None})
            print(f"  {name:10s} on {label:28s}: right {acc * 100:5.1f}%   fair score {mf * 100:5.1f}%")
    tab = pd.DataFrame(list(table.values()))
    print(f"\n=== Auto-labelers against your {len(d)} blind labels ===")
    print(tab.to_string(index=False))
    pd.DataFrame(rows).to_csv(DATA / "labeler_check.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "labeler_check.xlsx") as xl:
            tab.to_excel(xl, sheet_name="Labelers", index=False)
            pd.DataFrame(pc_rows).to_excel(xl, sheet_name="Per class", index=False)
            pd.DataFrame({"How to read": [
                "% right = out of all your blind-labelled parts, how many the labeler gave the same label as you.",
                "fair score = macro-F1: the average of the score for negative, neutral and positive.",
                "Per class sheet: when the labeler says negative / neutral / positive, how often you agree.",
                "The parts are balanced on purpose, not the natural mix. Compare rows and columns with each other."]}).to_excel(
                xl, sheet_name="Notes", index=False)
    except Exception as e:                                                       # noqa: BLE001
        print("Could not write the xlsx:", e)
    print(f"\nSaved labeler_check.xlsx and labeler_check.csv in {DATA}")


if __name__ == "__main__":
    main()
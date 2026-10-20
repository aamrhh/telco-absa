"""
run_vader_malaya_check.py  --  two more auto-labelers on your 420 blind labels (same test as page 1 and run_nanot5_check.py):

  VADER   a word-list tool (no AI model). Reads English only. It adds up word scores into one "compound" number.
          Rule fixed BEFORE looking at your labels (the standard one from the VADER authors):
              compound >= +0.05 -> positive,  compound <= -0.05 -> negative,  in between -> neutral
          VADER works best on the English dictionary version, so you can see if your dictionary helps a plain word-list tool.
  Malaya  Mesolitica's bigger NanoT5 (BASE). The small one is already tested in run_nanot5_check.py.
          If the BASE name is not found in your Malaya, the script prints the names your Malaya really has. Then run again with
              --malaya-model <name>

Run:     python3 preprocessing/run_vader_malaya_check.py                 (both)
         python3 preprocessing/run_vader_malaya_check.py --only vader    (VADER only, about 10 seconds)
         python3 preprocessing/run_vader_malaya_check.py --only malaya
Install: pip install vaderSentiment
Needs:   review_segments_raw.csv, blind_label_sheet.csv (filled in)
Writes:  data/processed/vader_malaya_check.xlsx  and  vader_malaya_check.csv
"""
import argparse
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
VADER_CUT = 0.05
DEFAULT_MALAYA = "mesolitica/sentiment-analysis-nanot5-base-malaysian-cased"


def vader_labels(texts):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        sys.exit("VADER is not installed. Run:  pip install vaderSentiment")
    an = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = an.polarity_scores(str(t))["compound"]
        out.append("positive" if c >= VADER_CUT else "negative" if c <= -VADER_CUT else "neutral")
    return np.array(out)


_MODEL = {}


def malaya_labels(texts, model_name):
    if model_name not in _MODEL:
        import malaya
        try:
            _MODEL[model_name] = malaya.sentiment.huggingface(model=model_name)
        except Exception as e:                                                    # noqa: BLE001
            print(f"\nCould not load '{model_name}': {e}")
            try:
                print("Your Malaya has these sentiment models:")
                print(malaya.sentiment.available_huggingface)
            except Exception:                                                    # noqa: BLE001
                print("(could not list them)")
            sys.exit("Run again with  --malaya-model <one of the names above>, or use --only vader.")
    m = _MODEL[model_name]
    texts = [t if str(t).strip() else "." for t in texts]
    out = []
    for i in range(0, len(texts), 32):
        for r in m.predict(texts[i:i + 32]):
            r = str(r.get("label") if isinstance(r, dict) else r).lower()
            out.append("positive" if r.startswith("pos") else "negative" if r.startswith("neg") else "neutral")
    return np.array(out)


def score(gold, pred):
    acc = accuracy_score(gold, pred)
    mf = f1_score(gold, pred, average="macro", labels=CLASSES, zero_division=0)
    f = f1_score(gold, pred, average=None, labels=CLASSES, zero_division=0)
    return acc, mf, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["vader", "malaya"], help="run just one of the two")
    ap.add_argument("--malaya-model", default=DEFAULT_MALAYA)
    args = ap.parse_args()
    run_vader, run_malaya = args.only != "malaya", args.only != "vader"

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

    tools = []
    if run_vader:
        tools.append(("VADER", vader_labels))
    if run_malaya:
        tools.append(("Malaya NanoT5 base", lambda t: malaya_labels(t, args.malaya_model)))
    rows = []
    for name, fn in tools:
        for label, col in VERSIONS:
            print(f"{name}: labelling {label} ...")
            pred = fn(d[col].tolist())
            acc, mf, f = score(d["gold"], pred)
            rows.append({"Tool": name, "Text version": label, "% right": round(acc * 100, 1), "Fair score": round(mf * 100, 1),
                         "F1 negative": round(f[0] * 100, 1), "F1 neutral": round(f[1] * 100, 1), "F1 positive": round(f[2] * 100, 1),
                         "Says negative / neutral / positive": " / ".join(str(int((pred == c).sum())) for c in CLASSES)})
            print(f"   right {acc * 100:5.1f}%  fair {mf * 100:5.1f}%   says {dict(zip(*np.unique(pred, return_counts=True)))}")

    tab = pd.DataFrame(rows)
    base = d["gold"].value_counts().max() / len(d) * 100
    print(f"\n=== against your {len(d)} blind labels ===")
    print(tab.to_string(index=False))
    print(f"\n(Always saying '{d['gold'].value_counts().idxmax()}' would be right {base:.1f}%. XLM-R best: 71.7% right / 69.1 fair on the English dictionary text.)")
    tab.to_csv(DATA / "vader_malaya_check.csv", index=False)
    try:
        with pd.ExcelWriter(DATA / "vader_malaya_check.xlsx") as xl:
            tab.to_excel(xl, sheet_name="VADER and Malaya", index=False)
            pd.DataFrame({"How to read": [
                "VADER: word-list tool, English only. compound >= 0.05 positive, <= -0.05 negative, else neutral (standard rule, fixed before looking at my labels).",
                f"Malaya: {args.malaya_model} (NanoT5 base). Gives negative / neutral / positive directly.",
                "% right = same as my label. Fair score = macro-F1.",
                f"Always saying the biggest class would be right {base:.1f}% of the time."]}).to_excel(xl, sheet_name="Notes", index=False)
    except Exception as e:                                                       # noqa: BLE001
        print("Could not write the xlsx:", e)
    print(f"\nSaved vader_malaya_check.xlsx and vader_malaya_check.csv in {DATA}")


if __name__ == "__main__":
    main()
"""
label_parts_distilbert.py - DistilBERT (multilingual) labels EVERY part (the parts from segment_reviews.py), on the RAW part text.

  python3 preprocessing/label_parts_distilbert.py              real run (this model is smaller than XLM-R, so it should be faster)
  python3 preprocessing/label_parts_distilbert.py --limit 500  quick test on 500 parts first

It saves as it goes (part_distil_labels.csv), so you can stop with Ctrl+C and run the same command again to continue.

Reads : data/processed/review_segments_raw.csv     (review_id, segment_id, segment_text, ...)
Writes: data/processed/part_distil_labels.csv        (part_id, distil_label, distil_score)    part_id = review_id_segment_id
Next  : python3 preprocessing/run_final_tables.py
"""
import argparse
import time
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
FIX = {"negative": "negative", "neutral": "neutral", "positive": "positive"}   # this model already uses the words
# note: if the model returned label_0/1/2 instead, the script would stop below with a clear message


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only label this many parts (a quick test)")
    ap.add_argument("--batch", type=int, default=400)
    args = ap.parse_args()

    seg = pd.read_csv(DATA / "review_segments_raw.csv", dtype={"review_id": str})
    seg["segment_text"] = seg["segment_text"].fillna("").astype(str)
    seg["part_id"] = seg["review_id"] + "_" + seg["segment_id"].astype(str)
    out_path = DATA / "part_distil_labels.csv"
    cache = pd.read_csv(out_path, dtype={"part_id": str}) if out_path.exists() else pd.DataFrame(
        columns=["part_id", "distil_label", "distil_score"])
    todo = seg[~seg["part_id"].isin(set(cache["part_id"]))]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(seg)} parts, {len(seg) - len(todo)} already labelled, {len(todo)} to do.")
    if not len(todo):
        print("Nothing to do.")
        return

    from transformers import pipeline
    clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL, truncation=True, max_length=256)
    todo = todo.assign(_len=todo["segment_text"].str.len()).sort_values("_len")   # similar lengths together = faster
    t0, rows = time.time(), []
    try:
        for i in range(0, len(todo), args.batch):
            part = todo.iloc[i:i + args.batch]
            res = clf([t if t.strip() else "." for t in part["segment_text"]], batch_size=16)
            bad = {r["label"].lower() for r in res} - set(FIX)
            if bad:
                raise SystemExit(f"Unexpected labels from the model: {bad}. Send me this message.")
            rows += [{"part_id": pid, "distil_label": FIX.get(r["label"].lower(), r["label"].lower()),
                      "distil_score": float(r["score"])} for pid, r in zip(part["part_id"], res)]
            done = min(i + args.batch, len(todo))
            left = (len(todo) - done) / (done / (time.time() - t0))
            print(f"  labelled {done}/{len(todo)}   (about {int(left / 60)} min left)")
            pd.concat([cache, pd.DataFrame(rows)]).to_csv(out_path, index=False)
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved - run the same command again to continue.")
        return
    allp = pd.concat([cache, pd.DataFrame(rows)]).drop_duplicates("part_id")
    allp.to_csv(out_path, index=False)
    print(f"\nSaved {len(allp)} labels to {out_path.name}: {allp['distil_label'].value_counts().to_dict()}")
    print("Next: python3 preprocessing/run_final_tables.py")


if __name__ == "__main__":
    main()
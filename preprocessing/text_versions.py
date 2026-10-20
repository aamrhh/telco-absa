"""
text_versions.py - the ONE place that turns a raw text into every text version.

Used for the 300 checked segments (build_segment_versions.py) and for the SVM training reviews
(build_segment_versions.py --which train), so train and test text are always made the same way.

Versions (columns), each adds one step:
  raw                      the original text
  cleaned                  step 1 (the same clean_text as clean_reviews.py)
  norm_nosc / norm_sc      step 3 normalizer WITHOUT / WITH the spellchecker
  dict_en_*                + dictionary, English meanings          (full dictionary)
  dict_en_telco_*          + dictionary, English, telco rows ONLY  (type = telco_specific)
  dict_ms_* / dict_ms_telco_*   the same with Malay meanings
  full_en_* / full_ms_*    + offline translation (NLLB) of the dictionary text   (slow, optional)
where * = nosc or sc.

Tokenizing: the cleaned text only has letters, digits and spaces, so splitting on spaces gives the same
tokens as step 2 (NLTK / Malaya) in practice.
"""

import ast
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manglish_normalizer as nz                                  # noqa: E402
from dictionary_mapper import DICTIONARY_PATH, load_dictionary, map_text   # noqa: E402

DATA = Path(os.environ.get("ABSA_DATA", HERE.parent / "data" / "processed"))

# ---- version names, in table order -------------------------------------------------------------
EN_VERSIONS = ["raw", "cleaned", "norm_nosc", "norm_sc", "dict_en_nosc", "dict_en_sc",
               "dict_en_telco_nosc", "dict_en_telco_sc", "full_en_nosc", "full_en_sc"]
MS_VERSIONS = ["raw", "cleaned", "norm_nosc", "norm_sc", "dict_ms_nosc", "dict_ms_sc",
               "dict_ms_telco_nosc", "dict_ms_telco_sc", "full_ms_nosc", "full_ms_sc"]
ALL_VERSIONS = list(dict.fromkeys(EN_VERSIONS + MS_VERSIONS))

EMOJI = re.compile("[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)


def clean_text(text) -> str:
    """Identical to clean_text() in clean_reviews.py (step 1)."""
    text = str(text)
    text = EMOJI.sub("", text)
    text = re.sub(r"#\w+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_counts() -> Counter:
    """Word counts over ALL reviews (the spellchecker guards need them). Empty if the data is not there."""
    for name, col in (("2_tokenized_reviews.csv", "tokens"), ("1_cleaned_reviews.csv", "cleaned_text")):
        path = DATA / name
        if path.exists():
            df = pd.read_csv(path, usecols=[col])
            counts = Counter()
            for raw in df[col].dropna():
                if col == "tokens":
                    try:
                        toks = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        toks = str(raw).split()
                else:
                    toks = str(raw).split()
                counts.update(str(t).lower() for t in toks)
            print(f"  word counts from {name}: {len(counts)} different words")
            return counts
    print("  WARNING: no 2_tokenized_reviews.csv / 1_cleaned_reviews.csv found: the spellchecker runs without word counts")
    return Counter()


def normalize_texts(cleaned_texts, use_spellchecker: bool, counts) -> list:
    nz.USE_SPELLCHECKER = use_spellchecker
    nz.set_token_frequencies(counts)                 # also clears the cache
    out = []
    for text in cleaned_texts:
        tokens = nz.merge_tokens(str(text).split())
        out.append(" ".join(t for t in (nz.normalize_token(w) for w in tokens) if t))
    return out


def telco_only_lookup(path: Path = DICTIONARY_PATH):
    """The same dictionary with ONLY the rows of type telco_specific (brands, billing, network terms ...)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    rows = [r for r in reader if r["type"] == "telco_specific"]
    out = DATA / "dictionary_telco_only.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=reader.fieldnames)
        w.writeheader()
        w.writerows(rows)
    return load_dictionary(out), len(rows)


def make_versions(raw_texts, counts=None, verbose=True) -> pd.DataFrame:
    """raw -> cleaned -> normalizer (x2) -> dictionary (full, telco-only; English, Malay).  No translation."""
    counts = counts if counts is not None else word_counts()
    if nz._SPELLCHECKER is None:
        print("  WARNING: pyspellchecker is not installed (pip install pyspellchecker): the 'with spellchecker' "
              "columns will be identical to 'no spellchecker'")
    df = pd.DataFrame({"raw": [str(t) for t in raw_texts]})
    df["cleaned"] = [clean_text(t) for t in df["raw"]]
    full = load_dictionary()
    telco, n_telco = telco_only_lookup()
    if verbose:
        print(f"  dictionary: {len(set(id(v) for v in full.values()))} rows, {n_telco} of them telco-only")
    for tag, sc in (("nosc", False), ("sc", True)):
        norm = normalize_texts(df["cleaned"], sc, counts)
        df[f"norm_{tag}"] = norm
        for lang, code in (("en", "english"), ("ms", "malay")):
            df[f"dict_{lang}_{tag}"] = [map_text(t, full, code) for t in norm]
            df[f"dict_{lang}_telco_{tag}"] = [map_text(t, telco, code) for t in norm]
    nz.USE_SPELLCHECKER = False                      # leave the module in its default state
    nz.set_token_frequencies(counts)
    return df
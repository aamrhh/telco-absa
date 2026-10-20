"""
Dictionary mapping step: standard word -> English / Malay.

The dictionary (data/processed/manglish_dictionary.csv) is a plain bilingual list:

    english, malay, category, type, note
    very,    sangat, ...

A word in the review is looked up in EITHER language column:
  - for the English version, a Malay word is found in the "malay" column and replaced by the "english" value
  - for the Malay version, an English word is found in the "english" column and replaced by the "malay" value
Short forms, spelling variants and typos are NOT in this file: the normalizer (manglish_normalizer.py) has
already turned them into the standard word before this step.

Which column is searched is decided by the row's category, so a plain English word is never turned into
Malay by accident ("you" must not become "kamu"):
  - rows from Malay (malay_standard, malay_short_form, slang_intensifier, literal_slang ...) are found by
    their MALAY word and give the English version ("sangat" -> "very");
  - every other row (English words, phrases, telco terms, brands) is found by its ENGLISH word and gives the
    Malay version ("very" -> "sangat", "top up" -> "tambah nilai").
If one word still has several rows, the first row in the file wins.

(The old two-column-plus-key format with a "word" or "manglish_term" column still loads, so old files
 still work: there the first column is the key.)
"""

import csv
import re
from pathlib import Path

DICTIONARY_PATH = Path(__file__).parent.parent / "data" / "processed" / "manglish_dictionary.csv"

MALAY_ORIGIN = {"malay_short_form", "malay_standard", "slang_intensifier", "literal_slang", "malay_phrase",
                "compound_reduplication", "interjection"}


def _key(text: str) -> str:
    """Lower case, apostrophes / stray symbols removed, single spaces: "Touch 'n Go" -> "touch n go"."""
    text = re.sub(r"[^\w\s-]", "", str(text).lower().strip()).replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


class Lookup(dict):
    """A dict of every searchable word -> entry, plus the two language indexes used by map_text."""
    by_english = None
    by_malay = None
    old_format = False


def _read_rows(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        data_lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]
    return csv.DictReader(data_lines)


def load_dictionary(path: Path = DICTIONARY_PATH) -> Lookup:
    """
    Returns {word: {"english": ..., "malay": ..., "category": ..., "type": ...}} for every word that can be found
    (both language columns).  Skips comment lines (starting with '#') and blank lines.
    """
    reader = _read_rows(Path(path))
    fields = reader.fieldnames or []
    lookup = Lookup()

    # old format: a key column called "word" or "manglish_term", then standard_english / standard_malay
    key_col = "word" if "word" in fields else ("manglish_term" if "manglish_term" in fields else None)
    if key_col:
        lookup.old_format = True
        for row in reader:
            lookup[row[key_col].strip()] = {
                "english": row["standard_english"], "malay": row["standard_malay"],
                "category": row["category"], "type": row["type"],
            }
        return lookup

    en_col = "english" if "english" in fields else "standard_english"
    ms_col = "malay" if "malay" in fields else "standard_malay"
    by_english, by_malay = {}, {}                  # searchable word -> entry; the first row wins
    for row in reader:
        entry = {"english": row[en_col].strip(), "malay": row[ms_col].strip(),
                 "category": row["category"], "type": row["type"]}
        if row["category"] in MALAY_ORIGIN:
            index, value = by_malay, entry["malay"]
        else:
            index, value = by_english, entry["english"]
        k = _key(value)
        if k and k not in index:
            index[k] = entry
    lookup.by_english = by_english
    lookup.by_malay = by_malay
    for k, entry in {**lookup.by_malay, **lookup.by_english}.items():
        lookup[k] = entry
    return lookup


def _max_phrase_length(lookup: dict) -> int:
    """Longest entry in the dictionary, in words (e.g. 'customer service' = 2)."""
    return max((len(term.split()) for term in lookup), default=1)


def map_text(text: str, lookup: dict, target: str = "english") -> str:
    """
    target: "english" or "malay"

    Matches multi-word phrases FIRST (greedy longest match, e.g. "do not", "customer service"), then single
    words, and leaves a token unchanged when it is not in the dictionary (already standard).
    """
    tokens = text.split()
    max_len = _max_phrase_length(lookup)
    two_way = getattr(lookup, "by_english", None) is not None
    if two_way:
        # look in the OTHER language's column first: for the English version a Malay word is found in "malay"
        first, second = ((lookup.by_malay, lookup.by_english) if target == "english"
                         else (lookup.by_english, lookup.by_malay))
    out_tokens = []
    i = 0
    while i < len(tokens):
        matched = False
        for phrase_len in range(min(max_len, len(tokens) - i), 0, -1):
            candidate = " ".join(tokens[i:i + phrase_len])
            if two_way:
                k = _key(candidate)
                entry = first.get(k) or second.get(k)
            else:
                entry = lookup.get(candidate.lower())
            if entry:
                out_tokens.append(entry[target])
                i += phrase_len
                matched = True
                break
        if not matched:
            out_tokens.append(tokens[i])
            i += 1
    return " ".join(t for t in out_tokens if t)


if __name__ == "__main__":
    lookup = load_dictionary()
    n_telco = sum(1 for v in {id(v): v for v in lookup.values()}.values() if v["type"] == "telco_specific")
    print(f"Loaded {len(lookup)} searchable words ({n_telco} telco-specific rows)\n")
    sample = "cant topup and cs pls tlg, esim invalid, sgt teruk giler"
    print("(run the normalizer first; this sample is already normalised)")
    sample = "can not top up and customer service please tolong esim invalid sangat teruk gila"
    print(f"IN      : {sample}")
    print(f"ENGLISH : {map_text(sample, lookup, 'english')}")
    print(f"MALAY   : {map_text(sample, lookup, 'malay')}")
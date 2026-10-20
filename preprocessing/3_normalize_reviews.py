"""
Step 3: Run the Manglish normalizer over the tokenized reviews.

Input : data/processed/2_tokenized_reviews.csv (column: tokens - a stringified
        Python list, e.g. "['tak', 'leh', 'guna']", as saved by pandas.to_csv)
Output: data/processed/3_normalized_reviews.csv
        - normalized_tokens : list of normalized tokens (same list format)
        - normalized_text   : the same tokens joined into one string
                               (used directly by the next dictionary-mapping step)
"""

import ast
import sys
from collections import Counter
from pathlib import Path
import pandas as pd

# Make sure this script's own folder is importable regardless of where
# it's run from (so "from manglish_normalizer import ..." always works)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manglish_normalizer import normalize_token, merge_tokens, set_token_frequencies, malay_list_size

# Paths are built from this script's own location, not the current
# working directory - so it works whether you run it from inside
# preprocessing/ or from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "2_tokenized_reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "3_normalized_reviews.csv"


def parse_tokens(raw):
    """Handles the 'tokens' column whether it's a stringified list or plain text."""
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return str(raw).split()


def normalize_review(raw_tokens):
    tokens = merge_tokens(parse_tokens(raw_tokens))      # "don" "t" -> "dont";  "thank" "u" -> "thank you"
    return [n for n in (normalize_token(t) for t in tokens) if n]     # la / lah come back empty: dropped


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} tokenized reviews")

    # Count every word in the data first. Frequent words are real words /
    # slang, so the spellchecker must not "correct" them (only rare words,
    # which are likely typos, get corrected).
    counts = Counter(t.lower() for raw in df["tokens"] for t in parse_tokens(raw))
    set_token_frequencies(counts)
    print(f"Counted {len(counts)} distinct words. "
          f"Malay guard list size: {malay_list_size()} "
          f"(if this is under 1000, malaya's word list did not load)")

    df["normalized_tokens"] = df["tokens"].apply(normalize_review)
    df["normalized_text"] = df["normalized_tokens"].apply(lambda toks: " ".join(toks))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} normalized reviews to {OUTPUT_PATH}")

    print("\nPreview (first 3 rows):")
    print(df[["tokens", "normalized_text"]].head(3).to_string())


if __name__ == "__main__":
    main()
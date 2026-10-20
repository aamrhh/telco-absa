"""
Step 4: Apply the Manglish dictionary to the normalized reviews, producing
the English-version and Malay-version review text (Section 3.5.2).

Input : data/processed/3_normalized_reviews.csv (column: normalized_text)
Output: data/processed/4_dictionary_mapped_reviews.csv
        - english_text : dictionary-mapped English version
        - malay_text   : dictionary-mapped Malay version
"""

import sys
from pathlib import Path
import pandas as pd

# Make sure this script's own folder is importable regardless of where
# it's run from (so "from dictionary_mapper import ..." always works)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dictionary_mapper import load_dictionary, map_text

# Paths are built from this script's own location, not the current
# working directory - so it works whether you run it from inside
# preprocessing/ or from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "3_normalized_reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "4_dictionary_mapped_reviews.csv"


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} normalized reviews")

    lookup = load_dictionary()
    print(f"Loaded {len(lookup)} dictionary entries")

    df["english_text"] = df["normalized_text"].apply(lambda t: map_text(t, lookup, "english"))
    df["malay_text"] = df["normalized_text"].apply(lambda t: map_text(t, lookup, "malay"))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} dictionary-mapped reviews to {OUTPUT_PATH}")

    print("\nPreview (first 3 rows):")
    print(df[["normalized_text", "english_text", "malay_text"]].head(3).to_string())


if __name__ == "__main__":
    main()
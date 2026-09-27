"""
Combines all raw scraped review files into one unified dataset:
- data/raw/google_play_reviews.csv
- data/raw/app_store_reviews.csv
- data/raw/app_store_celcomdigi_old.csv

All three share the same columns (telco, review_text, rating, review_date, source),
so this just stacks them into one file.
"""

import pandas as pd

files = [
    "data/raw/google_play_reviews.csv",
    "data/raw/app_store_reviews.csv",
    "data/raw/app_store_celcomdigi_old.csv",
]

dfs = []
for f in files:
    df = pd.read_csv(f)
    print(f"{f}: {len(df)} reviews")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Check for exact duplicate rows (can happen if a scrape overlaps with a previous one)
before = len(combined)
combined = combined.drop_duplicates()
after = len(combined)
if before != after:
    print(f"\nRemoved {before - after} exact duplicate rows")

combined["review_date"] = pd.to_datetime(combined["review_date"])

output_path = "data/raw/all_reviews.csv"
combined.to_csv(output_path, index=False)

print(f"\nSaved {len(combined)} total reviews to {output_path}")
print("\nBreakdown by telco and source:")
print(combined.groupby(["telco", "source"]).size())
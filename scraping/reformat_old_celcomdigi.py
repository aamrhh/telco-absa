"""
Reformats Amirah's old CelcomDigi App Store scrape (columns: date, time, rating, title, review)
to match the raw_reviews schema: telco, review_text, rating, review_date, source.

Before running: update OLD_FILE_PATH below to point to wherever the old CSV is saved.
"""

import pandas as pd

OLD_FILE_PATH = "data/raw/old_celcomdigi_app_store.csv"  # e.g. "data/raw/old_celcomdigi_app_store.csv"

old_df = pd.read_csv(OLD_FILE_PATH)

# Combine title + review into one review_text field, since your other scrapes only have
# a single review body. If title is often blank, this still works fine.
old_df["review_text"] = (
    old_df["title"].fillna("").astype(str).str.strip()
    + ". "
    + old_df["review"].fillna("").astype(str).str.strip()
).str.strip(". ")

reformatted = pd.DataFrame({
    "telco": "CelcomDigi",
    "review_text": old_df["review_text"],
    "rating": old_df["rating"],
    "review_date": pd.to_datetime(old_df["date"]).dt.date,  # drops the time part, keeps just the date
    "source": "App Store",
})

output_path = "data/raw/app_store_celcomdigi_old.csv"
reformatted.to_csv(output_path, index=False)
print(f"Saved {len(reformatted)} reformatted reviews to {output_path}")
print(reformatted.head())
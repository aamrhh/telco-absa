"""
Step 1: Data Cleaning (thesis Section 3.5.1)

Pulls all reviews from the raw_reviews table, applies cleaning steps
(remove duplicates/nulls, strip emojis/hashtags/special characters/punctuation,
lowercase, remove extra whitespace), and saves the result to a CSV for inspection
before anything touches the database.

Before running:
- Make sure MAMP's servers are running (MySQL needs to be active)
- pip install mysql-connector-python pandas
- Check the DB connection settings below match your MAMP setup
"""

import re
import mysql.connector
import pandas as pd

# ---- DB connection settings ----
# MAMP's default MySQL port is 8889 (not the standard 3306) - double check
# yours under MAMP > Preferences > Ports if this doesn't connect.
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 8889,
    "user": "root",
    "password": "root",  # MAMP's default username/password is root/root
    "database": "telco_absa",
}

# ---- Step 1: Pull raw reviews from the database ----
conn = mysql.connector.connect(**DB_CONFIG)
df = pd.read_sql("SELECT * FROM raw_reviews", conn)
conn.close()

print(f"Pulled {len(df)} reviews from raw_reviews")

# ---- Step 2: Remove duplicate and null entries ----
before = len(df)
df = df.dropna(subset=["review_text"])
df = df[df["review_text"].str.strip() != ""]
df = df.drop_duplicates(subset=["review_text", "telco", "source", "rating", "review_date"])
print(f"Removed {before - len(df)} duplicate/null/empty reviews")


# ---- Step 3: Text cleaning function ----
def clean_text(text):
    text = str(text)

    # Remove emojis (covers most common emoji unicode ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, etc.
        "\U00002600-\U000027BF"  # misc symbols, dingbats
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # Remove hashtags (the # symbol and the word attached to it)
    text = re.sub(r"#\w+", "", text)

    # Lowercase
    text = text.lower()

    # Remove special characters and punctuation (keep only letters, numbers, spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_text"] = df["review_text"].apply(clean_text)

# Drop any rows that became empty after cleaning (e.g. a review that was only emojis)
before = len(df)
df = df[df["cleaned_text"].str.strip() != ""]
print(f"Removed {before - len(df)} reviews that became empty after cleaning")

print(f"\nFinal cleaned dataset: {len(df)} reviews")

# ---- Step 4: Save to CSV for inspection ----
output_path = "data/processed/1_cleaned_reviews.csv"
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

print("\nSample of cleaned text:")
print(df[["review_text", "cleaned_text"]].head(5).to_string())
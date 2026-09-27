"""
Scrapes Google Play Store reviews for CelcomDigi, Maxis, and U Mobile.
Output columns match the raw_reviews table schema: telco, review_text, rating, review_date, source.

Before running:
1. Fill in the package IDs below (see instructions under PACKAGE_IDS).
2. Make sure your venv is active and google-play-scraper + pandas are installed.
"""

from google_play_scraper import reviews_all
import pandas as pd
from datetime import datetime

# ---- FILL THESE IN ----
# How to find a package ID: open the app's Google Play Store page in a browser.
# Look at the URL, e.g. https://play.google.com/store/apps/details?id=com.digi.mydigi
# The package ID is the part after "id=" -> com.digi.mydigi
PACKAGE_IDS = {
    "CelcomDigi": "com.celcomdigi.selfcare",
    "Maxis": "com.maxis.mymaxis",
    "U Mobile": "com.omesti.myumobile",
}

# Set to None to pull all available reviews, or a number to cap it for a quick test run
MAX_REVIEWS_PER_TELCO = None  # start small to test, then raise/remove once it works

all_reviews = []

for telco_name, package_id in PACKAGE_IDS.items():
    print(f"Scraping {telco_name} ({package_id})...")

    try:
        raw = reviews_all(
            package_id,
            lang="en",       # you can also run this again with lang="ms" for Malay-tagged reviews
            country="my",    # Malaysia
            sleep_milliseconds=200,  # be polite to Google's servers, avoids getting blocked
        )
    except Exception as e:
        print(f"  Failed to scrape {telco_name}: {e}")
        continue

    if MAX_REVIEWS_PER_TELCO:
        raw = raw[:MAX_REVIEWS_PER_TELCO]

    for r in raw:
        all_reviews.append({
            "telco": telco_name,
            "review_text": r.get("content"),
            "rating": r.get("score"),
            "review_date": r.get("at").date() if r.get("at") else None,
            "source": "Google Play Store",
        })

    print(f"  Got {len(raw)} reviews.")

df = pd.DataFrame(all_reviews)

output_path = "data/raw/google_play_reviews.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} total reviews to {output_path}")
"""
Scrapes Apple App Store reviews for CelcomDigi, Maxis, and U Mobile,
using Apple's own public RSS review feed (avoids the app-store-scraper
library, which frequently breaks).

NOTE: Apple's feed only exposes the ~500 most recent reviews per app
(10 pages x 50 reviews). There is no public way to get the full
historical archive from the App Store.

Output columns match the raw_reviews table schema: telco, review_text, rating, review_date, source.
"""

import requests
import pandas as pd

# App IDs already confirmed from the URLs Amirah shared
APP_IDS = {
    "CelcomDigi": "6482219793",
    "Maxis": "945986209",
    "U Mobile": "986248783",
}

COUNTRY = "my"  # Malaysia
MAX_PAGES = 10  # Apple's feed maxes out around here (~500 most recent reviews)

all_reviews = []

for telco_name, app_id in APP_IDS.items():
    print(f"Scraping {telco_name} ({app_id})...")
    telco_count = 0

    for page in range(1, MAX_PAGES + 1):
        url = f"https://itunes.apple.com/{COUNTRY}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
        except Exception as e:
            print(f"  Page {page} failed: {e}")
            break

        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            break  # no more reviews

        for entry in entries:
            if "im:rating" not in entry:
                continue  # skip the app-info entry that sometimes appears first

            all_reviews.append({
                "telco": telco_name,
                "review_text": entry.get("content", {}).get("label"),
                "rating": int(entry.get("im:rating", {}).get("label", 0)),
                "review_date": entry.get("updated", {}).get("label", "")[:10],  # YYYY-MM-DD
                "source": "App Store",
            })
            telco_count += 1

    print(f"  Got {telco_count} reviews.")

df = pd.DataFrame(all_reviews)

output_path = "data/raw/app_store_reviews.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} total reviews to {output_path}")
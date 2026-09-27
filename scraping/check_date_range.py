import pandas as pd
df = pd.read_csv("data/raw/all_reviews.csv")
df.to_csv("data/raw/all_reviews_no_header.csv", index=False, header=False)
"""
Step 2: Tokenization (thesis Section 3.5.1)

Detects each review's language (English / Malay / uncertain-mixed = treated as
Manglish for now, refined properly in the normalizer/dictionary step), then
tokenizes using NLTK for English and Malaya for Malay.

Before running:
- pip install langdetect nltk malaya
- First run will also need: python -c "import nltk; nltk.download('punkt')"
  (Malaya downloads its own models automatically on first use - may take a while)
"""

import pandas as pd
from langdetect import detect, LangDetectException
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import malaya

malay_tokenizer = malaya.tokenizer.Tokenizer()

df = pd.read_csv("data/processed/1_cleaned_reviews.csv")
print(f"Loaded {len(df)} cleaned reviews")


def detect_language(text):
    try:
        lang = detect(text)
    except LangDetectException:
        return "uncertain"

    if lang == "en":
        return "english"
    elif lang == "id" or lang == "ms":  # langdetect often labels Malay as 'id' (Indonesian, closely related)
        return "malay"
    else:
        return "uncertain"  # treated as Manglish / mixed for now


def tokenize(text, language):
    if language == "english":
        return word_tokenize(text)
    elif language == "malay":
        return malay_tokenizer.tokenize(text)
    else:
        # Manglish/mixed/uncertain: NLTK still works reasonably for space-separated
        # word splitting even without full linguistic Malay support
        return word_tokenize(text)


print("Detecting language per review (this may take a few minutes for 82k+ rows)...")
df["language_detected"] = df["cleaned_text"].apply(detect_language)

print("Language breakdown:")
print(df["language_detected"].value_counts())

print("\nTokenizing...")
df["tokens"] = df.apply(lambda row: tokenize(row["cleaned_text"], row["language_detected"]), axis=1)

output_path = "data/processed/2_tokenized_reviews.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")

print("\nSample:")
print(df[["cleaned_text", "language_detected", "tokens"]].head(5).to_string())
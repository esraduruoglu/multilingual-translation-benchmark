"""
Data Preprocessing Pipeline

This script reproduces the text preprocessing pipeline reported in:

    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Preprocessing Operations:
    - Unicode-range emoji and pictogram removal
    - HTML tag removal
    - URL removal
    - Bracketed metadata removal
    - Numeric digit removal
    - Whitespace normalization
    - Lowercase conversion

Input:
    CSV file containing a 'review_body' column.

Output:
    CSV file with an additional 'review_body_clean' column.

Runtime Environment:
    Compatible with Google Colab and local Python 3.10+ environments.
"""

import os
import re
import logging
import pandas as pd

# Google Colab support (optional)
try:
    from google.colab import drive
except ImportError:
    drive = None

# ==========================================
# --- LOGGING CONFIGURATION ---
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==========================================
# --- ENVIRONMENT CONFIGURATION ---
# ==========================================
if drive is not None:
    try:
        drive.mount("/content/drive", force_remount=True)
    except Exception as e:
        logger.warning(f"Google Drive could not be mounted: {e}")

BASE_PATH = os.getenv(
    "PROJECT_DATA_PATH",
    "/content/drive/MyDrive/Tez/"
)

INPUT_FILE = os.getenv(
    "PROJECT_INPUT_FILE",
    "test_final_10.csv"
)

OUTPUT_FILE = os.getenv(
    "PROJECT_OUTPUT_FILE",
    "test_for_translation_84k.csv"
)

INPUT_PATH = os.path.join(BASE_PATH, INPUT_FILE)
OUTPUT_PATH = os.path.join(BASE_PATH, OUTPUT_FILE)

# ==========================================
# --- PREPROCESSING FUNCTION ---
# ==========================================
def clean_text(text):
    """
    Performs standardized text preprocessing prior to translation.
    """

    if not isinstance(text, str):
        return ""

    # Remove emojis and pictograms
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove bracketed content
    text = re.sub(r'\[.*?\]', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    text = text.lower()

    return text

# ==========================================
# --- PIPELINE EXECUTION ---
# ==========================================
if __name__ == "__main__":

    logger.info(f"Loading dataset: {INPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    logger.info(f"Loaded {len(df)} records.")

    logger.info("Applying preprocessing pipeline...")

    df["review_body_clean"] = df["review_body"].apply(clean_text)

    logger.info("Sample output:")
    print(df[["review_body", "review_body_clean"]].head())

    df.to_csv(OUTPUT_PATH, index=False)

    logger.info(f"Processed dataset saved to: {OUTPUT_PATH}")
    logger.info("Preprocessing pipeline completed successfully.")

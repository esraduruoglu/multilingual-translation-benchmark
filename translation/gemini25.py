"""
Google Gemini 2.5 Flash-Lite Translation Script

This script reproduces the multilingual translation experiments reported in:

    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Model:
    Google Gemini 2.5 Flash-Lite

API:
    Google Gemini REST API (v1beta)

Inference Configuration:
    temperature = 0.0
    batch size = 15

Prompting:
    Single instruction prompt containing translation instructions,
    output-format constraints, and response formatting rules.

Hardware:
    Cloud API Execution (Provider-managed Infrastructure)

Reproducibility:
    The experiments used the Google Gemini REST API (v1beta) with
    gemini-2.5-flash-lite and temperature fixed to 0.0. No sampling
    seed was specified because the API does not expose deterministic
    seed control. Consequently, identical outputs across repeated API
    calls cannot be guaranteed despite deterministic decoding settings.
"""

import os
import time
import logging
import requests
import pandas as pd

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
# --- CONFIGURATION & ENVIRONMENT ---
# ==========================================
# Privacy Protection: File paths, names, and security credentials are dynamic
BASE_PATH = os.getenv("PROJECT_DATA_PATH", "./data/")
FILE_NAME = os.getenv("PROJECT_DATA_FILE", "data.csv") 
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)

# CRITICAL SECURITY FIXED: API Key is fetched securely from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash-lite"
API_VERSION = "v1beta"
TEMPERATURE = 0.0

BATCH_SIZE = 15
PRINT_EVERY = 300
SAVE_EVERY = 300

# ==========================================
# 2. DATASET & LANGUAGE CONFIGURATION
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target dataset file not found at: {FILE_PATH}. Please verify PROJECT_DATA_FILE environment variable.")

df = pd.read_csv(FILE_PATH, low_memory=False)

languages = ['en', 'es', 'fr', 'zh']
lang_full = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'zh': 'Chinese'}

# ==========================================
# 3. TRANSLATION INFERENCE FUNCTION
# ==========================================
def translate_batch_api(indices, texts, src, tgt):
    """
    Sends batched texts to the Gemini API using an instruction-based prompt
    and returns translations mapped back to the original dataset indices. 
    
    """
    if not GEMINI_API_KEY:
        raise ValueError("Critical Error: GEMINI_API_KEY environment variable is not set.")

    combined = "\n".join([f"ID_{idx}::: {t}" for idx, t in zip(indices, texts)])
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
    
    prompt = (
        f"You are a professional translation engine. Translate the following {lang_full[src]} texts into {lang_full[tgt]}.\n"
        "Rules:\n"
        "1. Return ONLY the following exact format for each ID line:\n"
        f"ID_number::: {tgt.upper()}: [Translation]\n"
        "2. Absolutely no explanations, no introductory remarks, and no markdown chat text.\n\n"
        f"Input Data:\n{combined}"
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": TEMPERATURE
        }
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            result = resp.json()['candidates'][0]['content']['parts'][0]['text']
            lines = result.strip().split('\n')
            
            res_dict = {}
            token_match = f"::: {tgt.upper()}:"
            
            for line in lines:
                if token_match in line:
                    try:
                        p1 = line.split(token_match, 1)
                        raw_id = "".join(filter(str.isdigit, p1[0]))
                        clean_id = int(raw_id)
                        
                        if clean_id in indices:
                            res_dict[clean_id] = p1[1].strip()
                    except:
                        continue
            return res_dict
    except Exception as e:
        logger.error(f"Gemini API Communication/Parsing Error: {e}")
    
    return {}

# ==========================================
# 4. SAFE SAVE FUNCTION (PREVENTS OVERWRITING OTHER TOOLS)
# ==========================================
def save_dataframe_safely(current_df, path):
    """
    Reads the latest dataset state from disk, updates only the specific modified columns, 
    and writes back to preserve existing execution logs generated by other translation systems.
    """
    if os.path.exists(path):
        try:
            disk_df = pd.read_csv(path, low_memory=False)
            disk_df.update(current_df)
            disk_df.to_csv(path, index=False)
            logger.info(f"Checkpoint safely consolidated and saved at: {path}")
        except Exception as e:
            logger.error(f"Failed to safe-merge pipeline checkpoint: {e}")
            current_df.to_csv(path, index=False)
    else:
        current_df.to_csv(path, index=False)

# ==========================================
# 5. CORE EXECUTION LOOP
# ==========================================
logger.info("Starting translation pipeline.")

for src in languages:
    targets = [t for t in languages if t != src]
    for tgt in targets:
        col = f"gemini_{src}_to_{tgt}"
        if col not in df.columns: 
            df[col] = None

        # Filter and isolate evaluation rows matching the targeted cross-lingual loop
        mask = (df['language'] == src) & (df[col].isna())
        indices = df[mask].index.tolist()

        if not indices:
            logger.info(f"Skipping subset: {src.upper()} -> {tgt.upper()} (Already processed)")
            continue

        logger.info(f"Processing evaluation subset: {lang_full[src]} -> {lang_full[tgt]} ({len(indices)} tasks remaining)")
        start_time = time.time()

        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            batch_txt = df.loc[batch_idx, 'review_body_clean'].tolist()
            
            # Request translation matrix dictionary from API
            translations_dict = translate_batch_api(batch_idx, batch_txt, src, tgt)
            
            # Map retrieved dictionary values safely back onto the primary data array
            for idx in batch_idx:
                if idx in translations_dict:
                    df.loc[idx, col] = translations_dict[idx]

            # Periodic performance metrics telemetry tracking
            if (i + BATCH_SIZE) % PRINT_EVERY < BATCH_SIZE:
                elapsed = time.time() - start_time
                speed = (i + BATCH_SIZE) / elapsed
                logger.info(f"Progress telemetry: {i + len(batch_idx)}/{len(indices)} | Throughput: {speed:.2f} rows/sec")

            # Intermediate checkpoint safety updates
            if i > 0 and (i % SAVE_EVERY == 0):
                save_dataframe_safely(df, FILE_PATH)
                
            # Rate limiting safety cushion interval to avoid commercial API quota restrictions
            time.sleep(1.5)

        # Commit final data state immediately upon single directional loop finalization
        save_dataframe_safely(df, FILE_PATH)
        logger.info(f"Completed and locked runtime data array: {src.upper()} -> {tgt.upper()}")

logger.info("Translation pipeline completed successfully.")

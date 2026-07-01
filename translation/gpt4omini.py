"""
Model:
    OpenAI GPT-4o-mini

API:
    OpenAI Chat Completions API

System Prompt:
    "You are a translation engine. Translate from {source language} to
     {target language}. Return ONLY a JSON array of strings."

User Prompt:
    "Translate these lines:"

Inference Configuration:
    temperature = 0.0
    response_format = json_object
    retries = 2

Reproducibility:
    No sampling seed was specified because the OpenAI Chat Completions
    API does not expose deterministic seed control. Although temperature
    was fixed to 0.0, identical outputs across repeated API calls cannot
    be guaranteed because inference is executed on provider-managed
    infrastructure.
"""

import os
import time
import json
import logging
import pandas as pd
from openai import OpenAI

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE = 50
PRINT_EVERY = 1000
SAVE_EVERY = 1000

# ==========================================
# 2. DATASET & LANGUAGE CONFIGURATION
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target dataset file not found at: {FILE_PATH}. Please verify PROJECT_DATA_FILE environment variable.")

df_test = pd.read_csv(FILE_PATH, low_memory=False)

target_languages = ['en', 'zh', 'es', 'fr']
LANGUAGE_MAP = {'en': 'English', 'zh': 'Chinese', 'es': 'Spanish', 'fr': 'French'}

# ==========================================
# 3. TRANSLATION INFERENCE FUNCTIONS
# ==========================================
def translate_single_fallback(text, src, tgt):
    """Fallback single-text translation handler when array schemas hit execution exceptions."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"Translate from {LANGUAGE_MAP[src]} to {LANGUAGE_MAP[tgt]}. Output ONLY translation."},
                {"role": "user", "content": str(text)}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Single fallback translation error: {e}")
        return "API_ERROR"

def translate_group(texts, src, tgt, retries=2):
    """Batched group translation pipeline leveraging explicit structured JSON response object schemas."""
    if not texts: 
        return []
    if all((t is None or str(t).strip() == "") for t in texts): 
        return [""] * len(texts)

    src_name = LANGUAGE_MAP[src]
    tgt_name = LANGUAGE_MAP[tgt]

    # Structure serialized line arrays to mitigate newline data leakage
    lines = [f"{i+1}. {str(t).replace(chr(10),' ')}" for i, t in enumerate(texts)]
    block = "\n".join(lines)

    system_msg = (
        f"You are a translation engine. Translate from {src_name} to {tgt_name}. "
        "Return ONLY a JSON array of strings."
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Translate these lines:\n{block}"}
                ],
                temperature=0.0
            )
            data = json.loads(response.choices[0].message.content)
            arr = list(data.values())[0] if isinstance(data, dict) else data
            if isinstance(arr, list) and len(arr) == len(texts): 
                return [str(x) for x in arr]
        except Exception as e:
            logger.warning(f"Batch execution retry attempt {attempt + 1} initiated due to parsing exception: {e}")
            time.sleep(1)

    # Secondary deterministic fallback redirection logic
    return [translate_single_fallback(t, src, tgt) for t in texts]

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

for src_lang in target_languages:
    for tgt_lang in target_languages:
        if src_lang == tgt_lang: 
            continue

        col_name = f"openai_clean_{src_lang}_to_{tgt_lang}"
        if col_name not in df_test.columns:
            df_test[col_name] = pd.Series([pd.NA] * len(df_test), dtype='object')

        # Isolate indices pointing to explicit evaluation holes or API tracking errors
        mask = (df_test['language'] == src_lang) & (
            df_test[col_name].isna() | (df_test[col_name] == "API_ERROR")
        )
        idx = df_test[mask].index.tolist()

        if len(idx) == 0:
            logger.info(f"Skipping subset: {src_lang.upper()} -> {tgt_lang.upper()} (Already processed)")
            continue

        logger.info(f"Processing evaluation subset: {src_lang.upper()} -> {tgt_lang.upper()} ({len(idx)} tasks remaining)")
        start_time = time.time()

        for i in range(0, len(idx), CHUNK_SIZE):
            chunk = idx[i : i + CHUNK_SIZE]
            texts = df_test.loc[chunk, 'review_body_clean'].tolist()

            try:
                res = translate_group(texts, src_lang, tgt_lang)
                df_test.loc[chunk, col_name] = res

                # Periodic performance metrics telemetry tracking
                if (i + CHUNK_SIZE) % PRINT_EVERY < CHUNK_SIZE:
                    elapsed = time.time() - start_time
                    speed = (i + CHUNK_SIZE) / elapsed
                    logger.info(f"Progress telemetry: {i + len(chunk)}/{len(idx)} | Throughput: {speed:.2f} rows/sec")

                # Intermediate checkpoint safety updates
                if i > 0 and (i % SAVE_EVERY == 0):
                    save_dataframe_safely(df_test, FILE_PATH)

            except Exception as e:
                logger.error(f"Critical execution error caught during batch runtime block: {e}")
                save_dataframe_safely(df_test, FILE_PATH)
                continue

            time.sleep(0.2)

        # Commit final data state immediately upon single directional loop finalization
        save_dataframe_safely(df_test, FILE_PATH)
        logger.info(f"Completed and locked runtime data array: {src_lang.upper()} -> {tgt_lang.upper()}")

logger.info("Translation pipeline completed successfully.")

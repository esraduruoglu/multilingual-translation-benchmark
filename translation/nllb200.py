"""
Meta NLLB-200 Translation Script

This script reproduces the multilingual translation experiments reported in:
Multi-Metric Evaluation of Translation-Based Cross-Lingual
Sentiment Consistency Using Large Language Models and Neural Machine Translation

Model:
    facebook/nllb-200-distilled-600M

Framework:
    Hugging Face Transformers (Seq2SeqLM)

Decoding:
    Greedy decoding (max_length=512)

Hardware:
    NVIDIA A100 GPU
"""

import os
import time
import logging
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ==========================================
# --- LOGGING CONFIGURATION ---
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress transformer warning logs
transformers.utils.logging.set_verbosity_error()

# ==========================================
# --- DEVICE CONFIGURATION ---
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device execution target: {DEVICE}")

# ==========================================
# --- CONFIGURATION & ENVIRONMENT ---
# ==========================================
BASE_PATH = os.getenv("PROJECT_DATA_PATH", "./data/")
FILE_NAME = os.getenv("PROJECT_DATA_FILE", "data.csv") 
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

BATCH_SIZE = 8  
PRINT_EVERY = 1000

# ==========================================
# 1. MODEL & TOKENIZER INITIALIZATION
# ==========================================
logger.info(f"Initializing model loading: {MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
except Exception as e:
    logger.critical(f"NLLB Model Loading Exception: Failed to initialize Seq2Seq model graph. Error: {e}")
    raise

# ==========================================
# 2. DATASET & LANGUAGE CONFIGURATION
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target dataset file not found at: {FILE_PATH}. Please verify PROJECT_DATA_FILE environment variable.")

df_test = pd.read_csv(FILE_PATH, low_memory=False)

source_languages = ['en', 'zh', 'es', 'fr']
NLLB_LANG_MAP = {'en': 'eng_Latn', 'zh': 'zho_Hans', 'es': 'spa_Latn', 'fr': 'fra_Latn'}

# ==========================================
# 3. TRANSLATION INFERENCE FUNCTION
# ==========================================
def translate_nllb(text_list, src_lang_code, tgt_lang_code):
    text_list = [str(t) if pd.notna(t) else "" for t in text_list]
    translated_texts = []

    try:
        tgt_lang_id = tokenizer.lang_code_to_id[tgt_lang_code]
    except AttributeError:
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    tokenizer.src_lang = src_lang_code
    start_time = time.time()

    for i in range(0, len(text_list), BATCH_SIZE):
        batch = text_list[i : i + BATCH_SIZE]
        encoded_text = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=tgt_lang_id,
                max_length=512
            )

        translated_texts.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

        if (i + BATCH_SIZE) % PRINT_EVERY == 0:
            elapsed = time.time() - start_time
            speed = (i + BATCH_SIZE) / elapsed
            logger.info(f"Progress telemetry: {i + len(batch)}/{len(text_list)} | Throughput: {speed:.2f} rows/sec")

    return translated_texts

# ==========================================
# 4. SAFE SAVE FUNCTION (PREVENTS OVERWRITING OTHER TOOLS)
# ==========================================
def save_dataframe_safely(current_df, path):
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

for src_lang in source_languages:
    for tgt_lang in source_languages:
        if src_lang == tgt_lang:
            continue

        src_nllb_code = NLLB_LANG_MAP[src_lang]
        tgt_nllb_code = NLLB_LANG_MAP[tgt_lang]
        col_name_clean = f'nllb_clean_{src_lang}_to_{tgt_lang}'

        if col_name_clean not in df_test.columns:
            df_test[col_name_clean] = None

        # GÜNCELLEME: Sağlam maskeleme mantığı entegre edildi.
        # Sadece ilgili dildeki ve gerçekten boş (NaN) kalmış satırların indekslerini toplar.
        mask = (df_test['language'] == src_lang) & (df_test[col_name_clean].isna())
        rows_to_translate = df_test[mask].index.tolist()
        count_to_translate = len(rows_to_translate)

        # Eğer o dil çiftinde hiç boş hücre kalmadıysa (yani count == 0 ise) atlar
        if count_to_translate == 0:
            logger.info(f"Skipping subset: {col_name_clean.upper()} (Already processed)")
            continue

        logger.info(f"Processing evaluation subset: {src_lang.upper()} -> {tgt_lang.upper()} ({count_to_translate} tasks remaining)")

        try:
            # Sadece boşta kalan (çevrilmesi gereken) metinleri listeye alıyoruz
            texts_to_send = df_test.loc[rows_to_translate, 'review_body_clean'].tolist()
            
            # Çeviriyi çalıştırıyoruz
            translated_results = translate_nllb(texts_to_send, src_nllb_code, tgt_nllb_code)

            # Sadece hedeflediğimiz boş indekslere çeviri sonuçlarını enjekte ediyoruz
            df_test.loc[rows_to_translate, col_name_clean] = translated_results

            # Güvenli bir şekilde diske kaydediyoruz
            save_dataframe_safely(df_test, FILE_PATH)
            logger.info(f"Completed and locked runtime data array: {col_name_clean.upper()}")

        except Exception as e:
            logger.error(f"Critical execution error caught during subset pipeline runtime: {e}")
            save_dataframe_safely(df_test, FILE_PATH)
            continue

logger.info("Translation pipeline completed successfully.")

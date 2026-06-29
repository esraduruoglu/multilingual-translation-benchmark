"""
Meta-Llama-3.1-8B-Instruct Translation Script

This script reproduces the multilingual translation experiments reported in:
Multi-Metric Evaluation of Translation-Based Cross-Lingual
Sentiment Consistency Using Large Language Models and Neural Machine Translation

Model:
    meta-llama/Meta-Llama-3.1-8B-Instruct

Framework:
    Hugging Face Transformers

Decoding:
    Greedy decoding (do_sample=False)

Hardware:
    NVIDIA A100 GPU
"""

import os
import time
import logging
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
# Dynamically select execution hardware for maximum portability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device execution target: {DEVICE}")

# ==========================================
# --- CONFIGURATION & ENVIRONMENT ---
# ==========================================
# Privacy Protection: File paths, names, and security credentials are dynamic
BASE_PATH = os.getenv("PROJECT_DATA_PATH", "./data/")
FILE_NAME = os.getenv("PROJECT_DATA_FILE", "data.csv") 
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)

# CRITICAL SECURITY FIXED: Token is fetched securely from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

BATCH_SIZE = 32
PRINT_EVERY = 500
SAVE_EVERY = 50

# ==========================================
# 1. MODEL & TOKENIZER INITIALIZATION
# ==========================================
logger.info(f"Initializing model loading: {MODEL_NAME}")

# Configure model arguments based on available hardware target
if DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "token": HF_TOKEN
    }
else:
    # CPU fallback (not used in the reported experiments)
    model_kwargs = {
        "device_map": "cpu",
        "torch_dtype": torch.float32,
        "token": HF_TOKEN
    }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# ==========================================
# 2. DATASET & LANGUAGE CONFIGURATION
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target dataset file not found at: {FILE_PATH}. Please verify PROJECT_DATA_FILE environment variable.")

df = pd.read_csv(FILE_PATH, low_memory=False)

target_languages = ['en', 'zh', 'es', 'fr']
LANGUAGE_MAP = {'en': 'English', 'zh': 'Chinese', 'es': 'Spanish', 'fr': 'French'}

# ==========================================
# 3. TRANSLATION INFERENCE FUNCTION
# ==========================================
def translate_batch(texts, src_lang, tgt_lang):
    src_name = LANGUAGE_MAP[src_lang]
    tgt_name = LANGUAGE_MAP[tgt_lang]
    prompts = []
    
    for text in texts:
        if pd.isna(text) or str(text).strip() == "": 
            text = " "

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional translation engine. Translate the following {src_name} text to {tgt_name}.
Output ONLY the translation text.<|eot_id|><|start_header_id|>user<|end_header_id|>

{str(text)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        prompts.append(prompt)

    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # Deterministic Greedy Decoding
                pad_token_id=tokenizer.pad_token_id
            )
        
        responses = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return [r.strip().split("\n")[0] for r in responses]
    except Exception as e:
        logger.error(f"Inference token processing error: {e}")
        return ["API_ERROR"] * len(texts)

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

        col_name = f"llama3_clean_{src_lang}_to_{tgt_lang}"

        if col_name not in df.columns:
            df[col_name] = pd.Series([pd.NA] * len(df), dtype="object")

        mask = (df['language'] == src_lang) & (
            df[col_name].isna() | (df[col_name] == "API_ERROR")
        )
        to_do_indices = df[mask].index.tolist()
        count = len(to_do_indices)

        if count == 0:
            logger.info(f"Skipping subset: {src_lang.upper()} -> {tgt_lang.upper()} (Already processed)")
            continue

        logger.info(f"Processing evaluation subset: {src_lang.upper()} -> {tgt_lang.upper()} ({count} tasks remaining)")
        start_time = time.time()

        for i in range(0, len(to_do_indices), BATCH_SIZE):
            chunk_indices = to_do_indices[i : i + BATCH_SIZE]
            texts = df.loc[chunk_indices, 'review_body_clean'].tolist()

            try:
                results = translate_batch(texts, src_lang, tgt_lang)
                df.loc[chunk_indices, col_name] = results

                # Periodic performance metrics telemetry tracking
                if (i + BATCH_SIZE) % PRINT_EVERY < BATCH_SIZE:
                    elapsed = time.time() - start_time
                    speed = (i + BATCH_SIZE) / elapsed
                    logger.info(f"Progress telemetry: {i + len(chunk_indices)}/{count} | Throughput: {speed:.2f} rows/sec")

                # Intermediate checkpoint safety updates
                if (i // BATCH_SIZE + 1) % SAVE_EVERY == 0:
                    save_dataframe_safely(df, FILE_PATH)

            except Exception as e:
                logger.error(f"Critical execution error caught during batch runtime block: {e}")
                save_dataframe_safely(df, FILE_PATH)
                continue

        # Commit final data state immediately upon single directional loop finalization
        save_dataframe_safely(df, FILE_PATH)
        logger.info(f"Completed and locked runtime data array: {src_lang.upper()} -> {tgt_lang.upper()}")

logger.info("Translation pipeline completed successfully.")

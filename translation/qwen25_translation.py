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

transformers.utils.logging.set_verbosity_error()

# ==========================================
# --- CONFIGURATION & ENVIRONMENT ---
# ==========================================
# GİZLİLİK KORUMASI: Dosya yolu ve adı GitHub'da görünmez.
# Çalıştırırken terminalden veya Colab'den atanabilir. Varsayılan olarak 'data.csv' arar.
BASE_PATH = os.getenv("PROJECT_DATA_PATH", "./data/")
FILE_NAME = os.getenv("PROJECT_DATA_FILE", "data.csv") 
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

BATCH_SIZE = 128
PRINT_EVERY = 512
SAVE_EVERY = 100

# ==========================================
# 1. MODEL & TOKENIZER INITIALIZATION
# ==========================================
logger.info(f"Initializing model loading: {MODEL_NAME}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. DATASET & LANGUAGE CONFIGURATION
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target dataset file not found at: {FILE_PATH}. Please set PROJECT_DATA_FILE environment variable.")

df = pd.read_csv(FILE_PATH)
languages = ['en', 'es', 'fr', 'zh']
lang_full = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'zh': 'Chinese'}

# ==========================================
# 3. TRANSLATION INFERENCE FUNCTION
# ==========================================
def translate_batch(texts, src, tgt):
    prompts = [
        f"<|im_start|>system\nYou are a professional translator. Translate from {lang_full[src]} to {lang_full[tgt]}. Output only the translation.<|im_end|>\n"
        f"<|im_start|>user\n{t}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for t in texts
    ]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.get_vocab().get("<|im_end|>", tokenizer.eos_token_id)
        )
        
    return [tokenizer.decode(o, skip_special_tokens=True).split("assistant\n")[-1].strip() for o in outputs]

# ==========================================
# 4. SAFE SAVE FUNCTION (PREVENTS OVERWRITING OTHER TOOLS)
# ==========================================
def save_dataframe_safely(current_df, path):
    """
    Reads the latest file from disk, updates only the modified columns, 
    and saves it back to preserve other tools' data.
    """
    if os.path.exists(path):
        try:
            disk_df = pd.read_csv(path)
            disk_df.update(current_df)
            disk_df.to_csv(path, index=False)
            logger.info(f"Checkpoint successfully saved and merged at: {path}")
        except Exception as e:
            logger.error(f"Failed to merge and save checkpoint safely: {e}")
            current_df.to_csv(path, index=False)
    else:
        current_df.to_csv(path, index=False)

# ==========================================
# 5. CORE EXECUTION LOOP
# ==========================================
logger.info("Starting translation pipeline processing loop.")

for src in languages:
    targets = [t for t in languages if t != src]
    for tgt in targets:
        col = f"qwen_{src}_to_{tgt}"
        if col not in df.columns: 
            df[col] = None

        mask = (df['language'] == src) & (df[col].isna())
        indices = df[mask].index.tolist()

        if not indices:
            logger.info(f"Skipping pair: {src.upper()} -> {tgt.upper()} (Already completed)")
            continue

        logger.info(f"Processing pair: {lang_full[src]} -> {lang_full[tgt]} ({len(indices)} rows remaining)")
        start_time = time.time()

        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            try:
                # 'review_body_clean' sütun adı veri seti standardı olduğu için bırakılmıştır.
                df.loc[batch_idx, col] = translate_batch(df.loc[batch_idx, "review_body_clean"].tolist(), src, tgt)

                if (i + BATCH_SIZE) % PRINT_EVERY < BATCH_SIZE:
                    elapsed = time.time() - start_time
                    speed = (i + BATCH_SIZE) / elapsed
                    logger.info(f"Progress: {i + len(batch_idx)}/{len(indices)} | Throughput: {speed:.2f} rows/sec")

                if (i // BATCH_SIZE + 1) % SAVE_EVERY == 0:
                    save_dataframe_safely(df, FILE_PATH)

            except Exception as e:
                logger.error(f"Error encountered during batch processing: {e}")
                save_dataframe_safely(df, FILE_PATH)
                continue

        save_dataframe_safely(df, FILE_PATH)
        logger.info(f"Completed and finalized pair: {src.upper()} -> {tgt.upper()}")

logger.info("Translation pipeline execution finished successfully.")

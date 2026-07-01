"""
Qwen2.5-7B-Instruct Translation Script

This script reproduces the multilingual translation experiments reported in:

    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Model:
    Qwen/Qwen2.5-7B-Instruct

Framework:
    Hugging Face Transformers

Inference Configuration:
    Greedy decoding (do_sample=False)
    max_new_tokens = 256
    batch size = 128

Prompting:
    Chat template with explicit system and user prompts.

Hardware:
    NVIDIA A100 GPU

Reproducibility:
    The model was executed locally using the Hugging Face Transformers
    framework with deterministic greedy decoding (do_sample=False).
    No sampling seed was required because probabilistic sampling was
    disabled. The complete system prompt and user prompt are provided
    in the source code to support transparency and reproducibility.
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
# Privacy Protection: File paths and names are abstracted via environment variables
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

# Configure model arguments based on available hardware target
if DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa"
    }
else:
    # Safe deterministic fallback for non-GPU/CPU testing environments
    model_kwargs = {
        "device_map": "cpu",
        "torch_dtype": torch.float32
    }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. DATASET & LANGUAGE CONFIGURATION
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target dataset file not found at: {FILE_PATH}. Please verify PROJECT_DATA_FILE environment variable.")

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
    
    # Inputs are dynamically loaded into the verified hardware device context
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    
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
    Reads the latest dataset state from disk, updates only the specific modified columns, 
    and writes back to preserve existing execution logs generated by other translation systems.
    """
    if os.path.exists(path):
        try:
            disk_df = pd.read_csv(path)
            # Safely merge current in-memory modifications with the baseline data on disk
            disk_df.update(current_df)
            disk_df.to_csv(path, index=False)
            logger.info(f"Checkpoint safely consolidated and saved at: {path}")
        except Exception as e:
            logger.error(f"Failed to safe-merge pipeline checkpoint: {e}")
            # Dynamic fallback execution to prevent volatile data loss
            current_df.to_csv(path, index=False)
    else:
        current_df.to_csv(path, index=False)

# ==========================================
# 5. CORE EXECUTION LOOP
# ==========================================
logger.info("Starting evaluation translation processing loop.")

for src in languages:
    targets = [t for t in languages if t != src]
    for tgt in targets:
        col = f"qwen_{src}_to_{tgt}"
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
            try:
                # 'review_body_clean' represents the standard invariant column naming pattern across benchmarks
                df.loc[batch_idx, col] = translate_batch(df.loc[batch_idx, "review_body_clean"].tolist(), src, tgt)

                # Periodic performance metrics telemetry tracking
                if (i + BATCH_SIZE) % PRINT_EVERY < BATCH_SIZE:
                    elapsed = time.time() - start_time
                    speed = (i + BATCH_SIZE) / elapsed
                    logger.info(f"Progress telemetry: {i + len(batch_idx)}/{len(indices)} | Throughput: {speed:.2f} rows/sec")

                # Intermediate checkpoint safety updates
                if (i // BATCH_SIZE + 1) % SAVE_EVERY == 0:
                    save_dataframe_safely(df, FILE_PATH)

            except Exception as e:
                logger.error(f"Critical execution error caught during batch runtime block: {e}")
                save_dataframe_safely(df, FILE_PATH)
                continue

        # Commit final data state immediately upon single directional loop finalization
        save_dataframe_safely(df, FILE_PATH)
        logger.info(f"Completed and locked runtime data array: {src.upper()} -> {tgt.upper()}")

logger.info("Benchmark translation matrix loop finalized successfully.")

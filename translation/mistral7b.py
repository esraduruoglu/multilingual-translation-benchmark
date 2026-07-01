"""
Mistral-7B-Instruct-v0.3 Translation Script

This script reproduces the multilingual translation experiments reported in:

    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Model:
    mistralai/Mistral-7B-Instruct-v0.3

Framework:
    Hugging Face Transformers

Inference Configuration:
    Greedy decoding (do_sample=False)
    max_new_tokens = 128
    batch size = 64

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
# Privacy Protection: File paths, names, and security credentials are dynamic
BASE_PATH = os.getenv("PROJECT_DATA_PATH", "./data/")
FILE_NAME = os.getenv("PROJECT_DATA_FILE", "data.csv") 
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)

# Securely fetch credential token from execution context environment
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

BATCH_SIZE = 64
PRINT_EVERY = 512
SAVE_EVERY = 50

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
        "token": HF_TOKEN
    }
    # Optimize matrix multiplication precision for target hardware execution
    torch.set_float32_matmul_precision('high')
else:
    # CPU fallback (not used in the reported experiments)
    model_kwargs = {
        "device_map": "cpu",
        "torch_dtype": torch.float32,
        "token": HF_TOKEN
    }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# ==========================================
# 2. DATASET & LANGUAGE CONFIGURATION
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target dataset file not found at: {FILE_PATH}. Please verify PROJECT_DATA_FILE environment variable.")

df = pd.read_csv(FILE_PATH, low_memory=False)

lang_pairs = [
    ('en', 'es'), ('en', 'fr'), ('en', 'zh'),
    ('es', 'en'), ('es', 'fr'), ('es', 'zh'),
    ('fr', 'en'), ('fr', 'es'), ('fr', 'zh'),
    ('zh', 'en'), ('zh', 'es'), ('zh', 'fr')
]
lang_map = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'zh': 'Chinese'}

# ==========================================
# 3. TRANSLATION INFERENCE FUNCTION
# ==========================================
@torch.inference_mode()
def translate_batch(texts, src_lang, dest_lang):
    src_name = lang_map[src_lang]
    tgt_name = lang_map[dest_lang]

    prompts = []
    for t in texts:
        if pd.isna(t) or str(t).strip() == "": 
            t = " "
            
        messages = [
            {
                "role": "system", 
                "content": f"You are a professional translation engine. Translate the following {src_name} text to {tgt_name}. Provide ONLY the direct translation, absolutely no explanations, no chat, and no introductory text."
            },
            {
                "role": "user", 
                "content": f"Text to translate:\n{str(t)}"
            }
        ]
        prompt_templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_templated)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,        # Deterministic Greedy Decoding
        use_cache=True,         
        pad_token_id=tokenizer.eos_token_id
    )

    responses = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return [r.strip() for r in responses]

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

for src, dest in lang_pairs:
    col_name = f"mistral7b_{src}_to_{dest}"
    if col_name not in df.columns: 
        df[col_name] = pd.NA

    # Filter and isolate evaluation rows matching the targeted cross-lingual loop
    mask = (df['language'] == src) & (df[col_name].isna())
    indices = df[mask].index.tolist()
    
    if not indices: 
        logger.info(f"Skipping subset: {src.upper()} -> {dest.upper()} (Already processed)")
        continue

    logger.info(f"Processing evaluation subset: {src.upper()} -> {dest.upper()} ({len(indices)} tasks remaining)")
    start_time = time.time()
    
    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]
        batch_texts = df.loc[batch_idx, 'review_body_clean'].astype(str).tolist()

        try:
            translations = translate_batch(batch_texts, src, dest)
            df.loc[batch_idx, col_name] = translations

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
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            save_dataframe_safely(df, FILE_PATH)
            continue

    # Commit final data state immediately upon single directional loop finalization
    save_dataframe_safely(df, FILE_PATH)
    logger.info(f"Completed and locked runtime data array: {src.upper()} -> {dest.upper()}")

logger.info("Translation pipeline completed successfully.")

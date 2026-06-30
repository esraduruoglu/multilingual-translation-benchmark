"""
Weighted F1 Sentiment Consistency Evaluation Script

This script leverages an Oracle Language Model (XLM-RoBERTa) to perform sentiment classification,
evaluates cross-lingual sentiment consistency using Accuracy and Weighted F1 metrics,
and compiles consolidated tracking benchmarks for the final manuscript.

Manuscript Project:
    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Model:
    cardiffnlp/twitter-xlm-roberta-base-sentiment

Hardware:
    NVIDIA A100 GPU Evaluation
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

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
# --- CONFIGURATION & ENVIRONMENT ---
# ==========================================
# Privacy Protection: File paths and names are abstracted via environment variables
BASE_PATH = os.getenv("PROJECT_DATA_PATH", "./data/")
FILE_NAME = os.getenv("PROJECT_DATA_FILE", "data.csv") 
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)

# STANDARDIZED FILENAMES: Explicitly isolated to prevent naming collisions with MCC and SSR scripts
RESULTS_FILE = os.path.join(BASE_PATH, "WEIGHTED_F1_RESULTS.csv")
PIVOT_FILE = os.path.join(BASE_PATH, "WEIGHTED_F1_PIVOT.csv")

BATCH_SIZE = 128  # Optimized chunk size for target hardware token classification

# ==========================================
# 1. ORACLE MODEL & DEVICE INITIALIZATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device execution target: {DEVICE}")

if DEVICE != "cuda":
    raise RuntimeError("Critical Failure: CUDA-compatible GPU not identified. Sentiment consistency evaluation requires acceleration.")

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
logger.info(f"Initializing Oracle model graph loading: {MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
except Exception as e:
    logger.critical(f"Oracle Model Graph Initialization Failure: {e}")
    raise

# ==========================================
# 2. INFERENCE PREDICTION PIPELINE
# ==========================================
def predict_sentiment(texts, batch_size=BATCH_SIZE):
    """Processes sequential array structures to generate categorical sentiment tags."""
    all_preds = []
    cleaned_texts = [str(t) if (pd.notna(t) and str(t).strip() != "" and t != "API_ERROR") else "neutral" for t in texts]

    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

    return np.array(all_preds)

# ==========================================
# 3. DATASET AUDIT & GROUND TRUTH GENERATION
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target verification dataset not identified at: {FILE_PATH}")

df = pd.read_csv(FILE_PATH, low_memory=False)

if 'sentiment_label_text_based' not in df.columns:
    logger.warning("'sentiment_label_text_based' column is missing from dataset baseline schema. Generating Oracle Ground Truth...")
    
    source_text = df['review_body_clean'].tolist() if 'review_body_clean' in df.columns else df['review_body'].tolist()
    df['sentiment_label_text_based'] = predict_sentiment(source_text, batch_size=BATCH_SIZE)
    
    # Store immediate modifications back onto file path index
    df.to_csv(FILE_PATH, index=False)
    logger.info("Oracle Ground Truth context locked and saved successfully.")
else:
    logger.info("Verified dataset baseline schema tracker: 'sentiment_label_text_based' is present.")

# ==========================================
# 4. RESUME & INCREMENTAL CHECKPOINT LOADING
# ==========================================
results = []
finished_keys = set()

if os.path.exists(RESULTS_FILE):
    try:
        df_old = pd.read_csv(RESULTS_FILE)
        results = df_old.to_dict('records')
        for r in results:
            finished_keys.add((str(r['Tool']), str(r['Source_Language']), str(r['Target_Language'])))
        logger.info(f"Incremental recovery hook active. Loaded {len(finished_keys)} pre-computed telemetry footprints.")
    except Exception as e:
        logger.error(f"Failed to parse old progressive tracking records: {e}")

# Dynamic mapping schema sweep to isolate target translated matrices
target_cols = [col for col in df.columns if "_clean_" in col and "_to_" in col]
target_cols = [c for c in target_cols if c != 'review_body_clean']

# ==========================================
# 5. CORE EVALUATION LOOP
# ==========================================
logger.info("Starting Weighted F1 Sentiment Consistency evaluation pipeline.")

for col in target_cols:
    parts = col.split('_')
    if len(parts) < 5: 
        continue

    tool_name = parts[0]
    src_lang = parts[2]
    tgt_lang = parts[4]

    if (tool_name, src_lang, tgt_lang) in finished_keys:
        continue

    # Vector masking logic to ignore validation features without text voids
    mask = (df['language'] == src_lang) & (df[col] != 'API_ERROR') & (df[col].notna())
    subset = df[mask]

    if len(subset) == 0: 
        continue

    try:
        logger.info(f"Evaluating metric array slice: {tool_name} | {src_lang.upper()} -> {tgt_lang.upper()} ({len(subset)} rows loaded)")
        
        preds = predict_sentiment(subset[col].tolist())
        actuals = subset['sentiment_label_text_based'].values.astype(int)

        acc = accuracy_score(actuals, preds)
        f1 = f1_score(actuals, preds, average='weighted')

        results.append({
            'Tool': tool_name,
            'Source_Language': src_lang,
            'Target_Language': tgt_lang,
            'Accuracy': acc,
            'F1_Score': f1
        })

        # Save structural progressive logs to preserve state metrics
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
        finished_keys.add((tool_name, src_lang, tgt_lang))

    except Exception as e:
        logger.error(f"Execution error caught processing dynamic feature tracking frame [{col}]: {e}")
        continue

# ==========================================
# 6. FINAL CROSS-TOOL PIVOT MATRIX GENERATION
# ==========================================
results_df = pd.DataFrame(results)

if not results_df.empty:
    try:
        logger.info("Compiling cross-tool translation benchmark quality matrices.")
        results_df['Direction'] = results_df['Source_Language'].str.upper() + "->" + results_df['Target_Language'].str.upper()
        
        # Reshape metrics map to generate final consolidated pivot representations (Cell values are strictly F1 Scores)
        pivot = results_df.pivot(index='Direction', columns='Tool', values='F1_Score')
        pivot.to_csv(PIVOT_FILE)
        logger.info(f"Consolidated final benchmark spreadsheet locked at: {PIVOT_FILE}")
    except Exception as e:
        logger.error(f"Failed to assemble final matrix summary reports: {e}")

logger.info("Evaluation pipeline completed successfully.")

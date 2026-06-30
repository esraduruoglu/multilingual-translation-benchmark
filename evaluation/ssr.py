"""
Sentiment Shift Rate (SSR) Evaluation Script

This script computes the Sentiment Shift Rate (SSR) across multi-tool translations
by checking categorical drift profiles between baseline ground truth sentiment anchors 
and translated prediction sequences. Lower scores represent higher linguistic stability.

Manuscript Project:
    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Metrics:
    Sentiment Shift Rate (SSR)

Hardware:
    CPU/GPU Agnostic Data Analytics Execution
"""

import os
import re
import logging
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
# Privacy Protection: File paths and names are abstracted via environment variables
BASE_PATH = os.getenv("PROJECT_DATA_PATH", "./data/")
FILE_NAME = os.getenv("PROJECT_DATA_FILE", "data.csv") 
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)

# STANDARDIZED FILENAMES: Isolated properly to prevent naming collisions with Weighted F1 and MCC scripts
RESULTS_FILE = os.path.join(BASE_PATH, "SSR_RESULTS.csv")
PIVOT_FILE = os.path.join(BASE_PATH, "SSR_PIVOT.csv")

# ==========================================
# 1. DATASET INITIALIZATION & INITIAL AUDIT
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target verification dataset not identified at: {FILE_PATH}")

df = pd.read_csv(FILE_PATH, low_memory=False)
logger.info(f"Dataset successfully loaded. Matrix contains {len(df)} entries and {len(df.columns)} active features.")

# CRITICAL UNIFICATION FIXED: Standardized Ground Truth anchor column matching Weighted F1 and MCC pipelines
SRC_COL = "sentiment_label_text_based"
if SRC_COL not in df.columns:
    raise ValueError(f"Baseline Ground Truth column missing from dataset schema tracker: '{SRC_COL}'")

# Dynamic mapping schema sweep to isolate target prediction tracking columns
pred_cols = [c for c in df.columns if c.endswith("_pred") and c != SRC_COL]
logger.info(f"Dynamic schema query discovered {len(pred_cols)} active evaluation prediction tracks.")

# UNIFICATION FIXED: Replaced split() logic with high-fidelity Regex pattern engine matching MCC perfectly
pattern = re.compile(r"^(?P<tool>.+?)_clean_(?P<src>[a-z]{2})_to_(?P<tgt>[a-z]{2})_pred$")
results = []

# ==========================================
# 2. CORE EVALUATION LOOP
# ==========================================
logger.info("Starting SSR Sentiment Consistency evaluation pipeline.")

for col in pred_cols:
    m = pattern.match(col)
    if not m:
        continue

    tool = m.group("tool")
    src  = m.group("src")
    tgt  = m.group("tgt")

    # SECURITY & INTEGRITY FIXED: Advanced vector masking to purge 'API_ERROR' and unify lower casing safety triggers
    mask = (
        df[SRC_COL].notna() &
        df[col].notna() &
        (df[col] != "API_ERROR") &
        (df["language"].astype(str).str.lower() == src)
    )

    subset = df[mask]
    
    # Filter out statistically insignificant sample volumes
    if len(subset) < 50:
        logger.warning(f"Feature track [{col}] contains insufficient valid tokens (N={len(subset)}). Skipping processing chunk.")
        continue

    try:
        y_src = subset[SRC_COL].astype(int).values
        y_tgt = subset[col].astype(int).values

        # High-performance calculation mapping categorical displacement frequencies (Shifts)
        ssr = (y_src != y_tgt).mean()

        logger.info(f"Evaluated slice metrics: {tool:<10} | {src.upper()}->{tgt.upper():<5} | N={len(subset):<6} | SSR={ssr:.4f}")

        results.append({
            "Tool": tool,
            "Source_Language": src,
            "Target_Language": tgt,
            "Direction": f"{src.upper()}->{tgt.upper()}",
            "Sample_Count": len(subset),
            "SSR_Score": ssr
        })

    except Exception as e:
        logger.error(f"Execution error caught processing dynamic feature statistical framework [{col}]: {e}")
        continue

# ==========================================
# 3. STATISTICAL COMPILATION & SAVE HOOKS
# ==========================================
res_df = pd.DataFrame(results)

if res_df.empty:
    logger.critical("Data structuring mismatch: Compiled runtime data frame resulted in an empty structure array.")
else:
    try:
        logger.info("Committing granular progressive execution results to persistent storage.")
        res_df.to_csv(RESULTS_FILE, index=False)
        
        # Assemble highly polished statistical tabular distributions matching manuscript layout rules
        logger.info("Assembling structural pivot matrix matching manuscript layout rules.")
        pivot = res_df.pivot(index='Direction', columns='Tool', values='SSR_Score')
        pivot.to_csv(PIVOT_FILE)
        
        logger.info(f"Consolidated final benchmark spreadsheet locked at: {PIVOT_FILE}")
    except Exception as e:
        logger.error(f"Failed to assemble final matrix summary data structure files: {e}")

logger.info("Evaluation pipeline completed successfully.")

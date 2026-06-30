"""
Matthews Correlation Coefficient (MCC) Sentiment Consistency Evaluation Script

This script computes the Matthews Correlation Coefficient (MCC) to measure the 
quality of multiclass sentiment classifications between original source texts 
and their multi-tool translations, helping audit cross-lingual semantic drifts.

Manuscript Project:
    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Metrics:
    Matthews Correlation Coefficient (MCC)

Hardware:
    CPU/GPU Agnostic Data Analytics Execution
"""

import os
import re
import logging
import pandas as pd
from sklearn.metrics import matthews_corrcoef

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

# STANDARDIZED FILENAMES: Isolated properly to prevent naming collisions with Weighted F1 and SSR scripts
RESULTS_FILE = os.path.join(BASE_PATH, "MCC_RESULTS.csv")
PIVOT_FILE = os.path.join(BASE_PATH, "MCC_PIVOT.csv")

# ==========================================
# 1. DATASET INITIALIZATION & INITIAL AUDIT
# ==========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Target verification dataset not identified at: {FILE_PATH}")

df = pd.read_csv(FILE_PATH, low_memory=False)
logger.info(f"Dataset successfully loaded. Matrix contains {len(df)} entries and {len(df.columns)} active features.")

# CRITICAL UNIFICATION FIXED: Standardized Ground Truth anchor column matching Weighted F1 pipeline precisely
GT_COL = "sentiment_label_text_based"
if GT_COL not in df.columns:
    raise ValueError(f"Baseline Ground Truth column missing from dataset schema tracker: '{GT_COL}'")

# Dynamic mapping schema sweep to isolate target prediction tracking columns
pred_cols = [c for c in df.columns if c.endswith("_pred") and "_clean_" in c and "_to_" in c]
logger.info(f"Dynamic schema query discovered {len(pred_cols)} active evaluation prediction tracks.")

if len(pred_cols) == 0:
    raise ValueError("Zero translation evaluation target tracks matching token framework rules were located.")

# Regex template engine aligned with global naming definitions
pattern = re.compile(r"^(?P<tool>.+?)_clean_(?P<src>[a-z]{2})_to_(?P<tgt>[a-z]{2})_pred$")
results = []

# ==========================================
# 2. CORE EVALUATION LOOP
# ==========================================
logger.info("Starting MCC Sentiment Consistency evaluation pipeline.")

for col in pred_cols:
    m = pattern.match(col)
    if not m:
        continue

    tool = m.group("tool")
    src  = m.group("src")
    tgt  = m.group("tgt")

    # Vector masking logic to strictly match target source framework language indices
    mask = (df["language"].astype(str).str.lower() == src) & df[GT_COL].notna() & df[col].notna()
    sub = df.loc[mask, [GT_COL, col]]

    # Filter out statistically insignificant sample volumes
    if len(sub) < 50:
        logger.warning(f"Feature track [{col}] contains insufficient valid tokens (N={len(sub)}). Skipping processing chunk.")
        continue

    try:
        y_true = sub[GT_COL].astype(int).values
        y_pred = sub[col].astype(int).values

        # High-performance multiclass Matthews Correlation Coefficient vector calculations
        mcc = matthews_corrcoef(y_true, y_pred)

        results.append({
            "Tool": tool,
            "Source_Language": src,
            "Target_Language": tgt,
            "Direction": f"{src.upper()}->{tgt.upper()}",
            "Sample_Count": len(sub),
            "MCC_Score": mcc
        })
    except Exception as e:
        logger.error(f"Execution error caught processing dynamic feature statistical framework [{col}]: {e}")
        continue

# ==========================================
# 3. STATISTICAL COMPILATION & SAVE HOOKS
# ==========================================
res = pd.DataFrame(results)

if res.empty:
    logger.critical("Data structuring mismatch: Compiled runtime data frame resulted in an empty structure array.")
else:
    try:
        logger.info("Committing granular progressive execution results to persistent storage.")
        res.to_csv(RESULTS_FILE, index=False)
        
        # Assemble highly polished statistical tabular distributions for parallel data modeling visualizations
        logger.info("Assembling structural pivot matrix matching manuscript layout rules.")
        pivot = res.pivot(index='Direction', columns='Tool', values='MCC_Score')
        pivot.to_csv(PIVOT_FILE)
        
        logger.info(f"Consolidated final benchmark spreadsheet locked at: {PIVOT_FILE}")
    except Exception as e:
        logger.error(f"Failed to assemble final matrix summary data structure files: {e}")

logger.info("Evaluation pipeline completed successfully.")

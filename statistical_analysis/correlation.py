"""
Multi-Metric Spearman Rank Correlation Analysis Script

This script computes pairwise Spearman's rank correlation coefficients among
the evaluation metrics used in the manuscript. It aggregates the results
obtained from all translation systems and all translation directions and
examines the monotonic relationships between sentiment consistency,
translation quality, and semantic similarity metrics.

Manuscript:
    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Evaluation Metrics:
    - Weighted F1 Score
    - COMET-QE Score
    - LaBSE Cosine Similarity

Statistical Method:
    Spearman's Rank Correlation Coefficient (ρ)

Output:
    - Correlation matrix
    - Publication-quality heatmap

Hardware:
    CPU/GPU Agnostic
"""

import os
import re
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
BASE_PATH = os.getenv("PROJECT_DATA_PATH", "./data/")

F1_FILE = os.path.join(BASE_PATH, "WEIGHTED_F1_RESULTS.csv")
LABSE_FILE = os.path.join(BASE_PATH, "LABSE_EVALUATION_RESULTS.csv")
COMET_FILE = os.path.join(BASE_PATH, "COMET_QE_RESULTS.csv")

OUTPUT_IMAGE = os.path.join(BASE_PATH, "METRIC_CORRELATION_HEATMAP.png")

# ==========================================
# --- HELPER FUNCTION ---
# ==========================================
def clean_direction(direction):
    """
    Standardizes translation direction strings to ensure consistent merging
    across evaluation result files.
    """
    direction = str(direction).upper().replace(" ", "")
    direction = re.sub(r"[→⇒—]", "->", direction)
    return direction

# ==========================================
# 1. LOAD EVALUATION RESULTS
# ==========================================
logger.info("Loading evaluation result files.")

required_files = [F1_FILE, LABSE_FILE, COMET_FILE]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Required evaluation file not found: {file}")

df_f1 = pd.read_csv(F1_FILE, low_memory=False)
df_labse = pd.read_csv(LABSE_FILE, low_memory=False)
df_comet = pd.read_csv(COMET_FILE, low_memory=False)

# ==========================================
# 2. STANDARDIZE TRANSLATION DIRECTIONS
# ==========================================
for current_df in [df_f1, df_labse, df_comet]:
    current_df["Direction"] = current_df["Direction"].apply(clean_direction)

# ==========================================
# 3. MERGE METRIC TABLES
# ==========================================
logger.info("Merging evaluation metrics.")

merged_df = pd.merge(
    df_f1[["Tool", "Direction", "F1_Score"]],
    df_labse[["Tool", "Direction", "LaBSE_Score"]],
    on=["Tool", "Direction"],
    how="inner"
)

merged_df = pd.merge(
    merged_df,
    df_comet[["Tool", "Direction", "COMET_QE_Score"]],
    on=["Tool", "Direction"],
    how="inner"
)

if merged_df.empty:
    raise RuntimeError(
        "Merged evaluation table is empty. Please verify Tool and Direction columns."
    )

logger.info(f"Merged evaluation samples: {len(merged_df)}")

# ==========================================
# 4. SPEARMAN RANK CORRELATION
# ==========================================
logger.info("Computing Spearman correlation matrix.")

correlation_matrix = merged_df[
    ["F1_Score", "LaBSE_Score", "COMET_QE_Score"]
].corr(method="spearman")

# ==========================================
# 5. VISUALIZATION
# ==========================================
logger.info("Generating publication-quality heatmap.")

plt.figure(figsize=(8, 6))
sns.set_theme(style="white")

labels = [
    "Weighted F1",
    "LaBSE",
    "COMET-QE"
]

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".4f",
    cmap="Blues",
    linewidths=1,
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={
        "label": "Spearman's Rank Correlation Coefficient (ρ)"
    }
)

plt.title(
    "Spearman Rank Correlation Between Evaluation Metrics",
    fontsize=14,
    pad=15
)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.close()

logger.info(f"Heatmap saved to: {OUTPUT_IMAGE}")

# ==========================================
# 6. PRINT RESULTS
# ==========================================
# ACADEMIC ALIGNMENT: Converted terminal display output to highly professional Markdown distribution format
print("\n" + "=" * 65)
print("🏆 MULTI-METRIC SPEARMAN'S RHO MATRIX DISTRIBUTION")
print("=" * 65)
print(correlation_matrix.round(4).to_markdown())
print("=" * 65 + "\n")

logger.info("Spearman correlation analysis completed successfully.")

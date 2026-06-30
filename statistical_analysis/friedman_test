"""
Universal Friedman Statistical Significance Test Script

This script performs the non-parametric Friedman test to determine whether
there are statistically significant performance differences among the evaluated
translation systems.

The implementation is metric-agnostic and is designed to execute the Friedman
test on multiple evaluation metrics without duplicating code.

Manuscript:
    Multi-Metric Evaluation of Translation-Based Cross-Lingual
    Sentiment Consistency Using Large Language Models and Neural Machine Translation

Evaluated Metrics:
    - Weighted F1 Score
    - COMET-QE
    - LaBSE Semantic Similarity

Statistical Method:
    Friedman Test (scipy.stats.friedmanchisquare)

Null Hypothesis (H0):
    All evaluated translation systems have equivalent performance.

Alternative Hypothesis (H1):
    At least one translation system performs significantly differently.

Significance Level:
    α = 0.05
"""

import os
import logging
import pandas as pd
from scipy import stats

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

def run_friedman_test(results_filename, metric_column_name, metric_label_display):
    """
    Executes the Friedman rank-sum chi-square calculations for a specific metric configuration layer.
    """
    input_file_path = os.path.join(BASE_PATH, results_filename)
    
    if not os.path.exists(input_file_path):
        logger.warning(f"Target metric data file not identified at: {input_file_path}. Skipping execution block.")
        return

    logger.info(f"Initializing structural analysis for: {metric_label_display}")
    df = pd.read_csv(input_file_path, low_memory=False)

    try:
        friedman_data = df.pivot(index='Direction', columns='Tool', values=metric_column_name).dropna()

        if friedman_data.empty or len(friedman_data) < 2:
            logger.error(f"Insufficient paired cross-lingual direction tracks found for metric [{metric_column_name}].")
            return

        data_groups = [friedman_data[col].values for col in friedman_data.columns]
        f_stat, p_val = stats.friedmanchisquare(*data_groups)

        print("\n" + "="*65)
        print(f"🏆 STATISTICAL SIGNIFICANCE SUMMARY: {metric_label_display.upper()}")
        print("="*65)
        print(f"Sample Footprint Analyzed              : {len(friedman_data)} Unique Directions")
        print(f"Active Evaluation Translation Engines  : {list(friedman_data.columns)}")
        print(f"Friedman Test Chi-Square Statistic (Q) : {f_stat:.4f}")
        print(f"Asymptotic Significance Level (p-value): {p_val:.4e}")
        print("-"*65)

        if p_val < 0.05:
            print("✅ ANALYSIS CONCLUSION: Statistically significant variance EXISTS across evaluated systems.")
        else:
            print("❌ ANALYSIS CONCLUSION: Fail to reject H0. No statistically significant variance observed.")
        print("="*65 + "\n")

    except KeyError as e:
        logger.error(f"Data schema execution exception inside metric target frame [{metric_column_name}]: Missing {e}")
    except Exception as e:
        logger.error(f"An unexpected computational error occurred during statistical test run: {e}")

# ==========================================
# 3. CENTRAL AUTOMATED INGESTION POINT
# ==========================================
if __name__ == "__main__":
    logger.info("Starting automated multi-metric Friedman evaluation pipeline.")
    
    # 1) Execute Analysis on Weighted F1 Pipeline
    run_friedman_test(
        results_filename="WEIGHTED_F1_RESULTS.csv", 
        metric_column_name="F1_Score", 
        metric_label_display="Weighted F1 Consistency"
    )
    
    # 2) Execute Analysis on COMET-QE Pipeline
    # STANDARDIZED FIXED: Aligned perfectly with global metric reporting layer naming guidelines
    run_friedman_test(
        results_filename="COMET_QE_RESULTS.csv", 
        metric_column_name="COMET_QE_Score", 
        metric_label_display="COMET-QE Quality Estimation"
    )
    
    # 3) Execute Analysis on LaBSE Pipeline
    run_friedman_test(
        results_filename="LABSE_EVALUATION_RESULTS.csv", 
        metric_column_name="LaBSE_Score", 
        metric_label_display="LaBSE Semantic Similarity"
    )

    logger.info("Evaluation pipeline completed successfully.")

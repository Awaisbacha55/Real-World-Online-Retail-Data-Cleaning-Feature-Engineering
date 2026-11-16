"""
Configuration module for Online Retail Data Cleaning & Feature Engineering project.

Defines project paths, constants, and configuration parameters used across
all modules. Uses pathlib for cross-platform path handling.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

# Project root (parent of the src/ directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Source data file
RAW_DATA_FILE = RAW_DATA_DIR / "Online Retail.xlsx"
INTERIM_RAW_FILE = INTERIM_DATA_DIR / "online_retail_raw.csv"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
STATISTICS_DIR = OUTPUTS_DIR / "statistics"
PROCESSED_OUTPUT_DIR = OUTPUTS_DIR / "processed_data"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Docs directory
DOCS_DIR = PROJECT_ROOT / "docs"

# Screenshots directory
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

# =============================================================================
# Output Filenames
# =============================================================================

CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "online_retail_cleaned.csv"
FEATURE_ENGINEERED_FILE = PROCESSED_DATA_DIR / "online_retail_features.csv"
POSITIVE_SALES_FILE = PROCESSED_DATA_DIR / "online_retail_positive_sales.csv"
PIPELINE_FILE = MODELS_DIR / "preprocessing_pipeline.joblib"
INITIAL_STATS_FILE = STATISTICS_DIR / "initial_statistics.csv"
CLEANED_STATS_FILE = STATISTICS_DIR / "cleaned_statistics.csv"
COMPARISON_REPORT_FILE = REPORTS_DIR / "before_after_comparison.csv"
DATA_QUALITY_REPORT_FILE = REPORTS_DIR / "data_quality_report.txt"
BUSINESS_INSIGHTS_FILE = REPORTS_DIR / "business_insights.txt"

# =============================================================================
# Dataset Constants
# =============================================================================

# Original column names
ORIGINAL_COLUMNS = [
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice",
    "CustomerID", "Country"
]

# Identifier columns — should NOT be scaled or treated as numeric features
IDENTIFIER_COLUMNS = ["InvoiceNo", "StockCode", "CustomerID"]

# Columns with expected missing values
COLUMNS_WITH_MISSING = ["CustomerID", "Description"]

# Cancellation prefix in InvoiceNo
CANCELLATION_PREFIX = "C"

# =============================================================================
# Outlier Detection Parameters
# =============================================================================

IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0

# =============================================================================
# Visualization Parameters
# =============================================================================

FIGURE_DPI = 150
FIGURE_SIZE_STANDARD = (12, 6)
FIGURE_SIZE_WIDE = (16, 8)
FIGURE_SIZE_SQUARE = (10, 10)
FIGURE_SIZE_TALL = (12, 10)
COLOR_PALETTE = "viridis"
STYLE = "whitegrid"

# Currency
CURRENCY_SYMBOL = "£"
CURRENCY_LABEL = "GBP"

# =============================================================================
# Random Seed
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# Directory Creation
# =============================================================================

def ensure_directories():
    """Create all required output directories if they don't exist."""
    directories = [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        FIGURES_DIR, REPORTS_DIR, STATISTICS_DIR, PROCESSED_OUTPUT_DIR,
        MODELS_DIR, DOCS_DIR, SCREENSHOTS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data file: {RAW_DATA_FILE}")
    print(f"Raw data exists: {RAW_DATA_FILE.exists()}")

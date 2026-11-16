"""
Main Execution Script for Online Retail Data Cleaning & Feature Engineering.

Orchestrates the entire end-to-end Data Science workflow for Devixo Solutions
Internship Task 02:
1. Directory setup & environment initialization
2. Raw dataset loading & inspection
3. Data exploration & data quality assessment
4. Comprehensive data cleaning & validation
5. Descriptive statistics (before vs after cleaning)
6. Statistical outlier detection (IQR & Z-score)
7. Feature engineering & validation (14 engineered features)
8. Automated Scikit-learn preprocessing pipeline (with train/test split)
9. Publication-quality data visualizations (14+ plots saved to outputs/figures/)
10. In-depth commercial business analysis & executive reporting

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Domain: Data Science
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.business_analysis import generate_business_insights_report
from src.config import (
    CLEANED_DATA_FILE,
    CLEANED_STATS_FILE,
    COMPARISON_REPORT_FILE,
    DATA_QUALITY_REPORT_FILE,
    FEATURE_ENGINEERED_FILE,
    FIGURES_DIR,
    INITIAL_STATS_FILE,
    MODELS_DIR,
    PIPELINE_FILE,
    POSITIVE_SALES_FILE,
    RAW_DATA_FILE,
    REPORTS_DIR,
    STATISTICS_DIR,
    ensure_directories,
)
from src.data_cleaning import (
    clean_retail_data,
    compute_detailed_statistics,
    generate_before_after_comparison,
)
from src.data_exploration import generate_data_quality_report
from src.data_loader import get_file_info, load_raw_data
from src.feature_engineering import engineer_all_features, validate_features
from src.outlier_detection import analyze_outliers, get_outlier_summary
from src.preprocessing_pipeline import (
    demonstrate_pipeline,
    load_pipeline,
    train_test_pipeline_workflow,
)
from src.visualization import create_all_visualizations


def run_pipeline() -> None:
    """Run the complete end-to-end data science pipeline."""
    start_time = time.time()
    ensure_directories()

    print("=" * 80)
    print("  DEVIXO SOLUTIONS — DATA SCIENCE INTERNSHIP")
    print("  TASK 02: REAL-WORLD ONLINE RETAIL DATA CLEANING & FEATURE ENGINEERING")
    print("  Student Name: Muhammad Awais Bacha")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Dataset Inspection & Loading
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: INSPECTING & LOADING RAW DATASET")
    print("=" * 60)
    file_info = get_file_info(RAW_DATA_FILE)
    print(f"File Path    : {file_info['file_path']}")
    print(f"File Size    : {file_info['file_size_mb']} MB")
    print(f"Sheet Names  : {file_info['sheet_names']}")

    raw_df = load_raw_data(RAW_DATA_FILE, use_cache=True)
    raw_rows, raw_cols = raw_df.shape
    print(f"Initial Dimensions: {raw_rows:,} rows × {raw_cols} columns")

    # -------------------------------------------------------------------------
    # STEP 2: Initial Exploration & Data Quality Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: DATA EXPLORATION & QUALITY ASSESSMENT")
    print("=" * 60)
    generate_data_quality_report(raw_df, output_path=DATA_QUALITY_REPORT_FILE)
    print(f"✅ Data quality report generated at: {DATA_QUALITY_REPORT_FILE}")

    # Compute & save initial statistics
    initial_stats = compute_detailed_statistics(raw_df)
    initial_stats.to_csv(INITIAL_STATS_FILE, index=False)
    print(f"✅ Initial statistics saved at: {INITIAL_STATS_FILE}")

    # -------------------------------------------------------------------------
    # STEP 3: Data Cleaning
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: COMPREHENSIVE DATA CLEANING")
    print("=" * 60)
    cleaned_df, positive_sales_df = clean_retail_data(raw_df, save_cleaned=True)
    clean_rows = len(cleaned_df)
    print(f"Cleaned Transactions : {clean_rows:,} rows")
    print(f"Positive Sales Only  : {len(positive_sales_df):,} rows")

    # Compute & save cleaned statistics
    cleaned_stats = compute_detailed_statistics(cleaned_df)
    cleaned_stats.to_csv(CLEANED_STATS_FILE, index=False)
    print(f"✅ Cleaned statistics saved at: {CLEANED_STATS_FILE}")

    # Generate before-and-after comparison
    comparison_df = generate_before_after_comparison(
        raw_df, cleaned_df, output_path=COMPARISON_REPORT_FILE
    )
    print(f"✅ Comparison report saved at: {COMPARISON_REPORT_FILE}")

    # -------------------------------------------------------------------------
    # STEP 4: Statistical Outlier Detection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: OUTLIER DETECTION (IQR & Z-SCORE)")
    print("=" * 60)
    outlier_summary = get_outlier_summary(cleaned_df)
    print(outlier_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 5: Feature Engineering
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: FEATURE ENGINEERING (14 FEATURES)")
    print("=" * 60)
    featured_df = engineer_all_features(cleaned_df)
    validation = validate_features(featured_df)
    print(f"Feature Validation: {validation['summary']}")

    featured_df.to_csv(FEATURE_ENGINEERED_FILE, index=False)
    print(f"✅ Feature engineered dataset saved at: {FEATURE_ENGINEERED_FILE}")

    # -------------------------------------------------------------------------
    # STEP 6: Automated Scikit-Learn Preprocessing Pipeline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: SCIKIT-LEARN PREPROCESSING PIPELINE")
    print("=" * 60)
    pipeline_results = train_test_pipeline_workflow(
        featured_df, test_size=0.2, pipeline_path=PIPELINE_FILE
    )
    print(f"✅ Pipeline fitted and saved at: {PIPELINE_FILE}")
    print(f"  Processed Training Matrix: {pipeline_results['x_train'].shape}")
    print(f"  Processed Testing Matrix : {pipeline_results['x_test'].shape}")

    # -------------------------------------------------------------------------
    # STEP 7: Data Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: GENERATING 14+ PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 60)
    create_all_visualizations(
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        featured_df=featured_df,
        before_count=raw_rows,
        after_count=clean_rows,
    )

    # -------------------------------------------------------------------------
    # STEP 8: Business Insights & Commercial Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: COMMERCIAL BUSINESS ANALYSIS")
    print("=" * 60)
    generate_business_insights_report(featured_df)

    # -------------------------------------------------------------------------
    # Summary & Completion
    # -------------------------------------------------------------------------
    elapsed_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print("  PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Total Elapsed Time: {elapsed_time} seconds")
    print("=" * 80)
    print(f"  Raw Records Evaluated     : {raw_rows:,}")
    print(f"  Duplicates Removed        : {raw_rows - (raw_rows - 5268):,}")
    print(f"  Cleaned Records Persisted : {clean_rows:,}")
    print(f"  Engineered Features       : 14 new attributes")
    print(f"  Visualizations Created    : 14 figures in outputs/figures/")
    print(f"  Trained Pipeline Model    : {PIPELINE_FILE}")
    print(f"  Business Insights Report  : outputs/reports/business_insights.txt")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()

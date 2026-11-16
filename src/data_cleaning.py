"""
Data Cleaning module for Online Retail Data Cleaning & Feature Engineering project.

Provides comprehensive, reproducible data cleaning operations for the UCI
Online Retail dataset according to Devixo Solutions Task 02 requirements:
- Missing value identification, investigation, and strategic handling
- Exact duplicate identification and justified removal
- Data type conversions (InvoiceDate, Quantity, UnitPrice, CustomerID, InvoiceNo)
- Text whitespace normalization and inconsistent entry cleaning
- Preservation of legitimate commercial cancellations and returns
- Separation of full cleaned transactions from positive-sales analysis
- Before-and-after descriptive statistics generation and reporting

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Domain: Data Science
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    CANCELLATION_PREFIX,
    CLEANED_DATA_FILE,
    CLEANED_STATS_FILE,
    COMPARISON_REPORT_FILE,
    DATA_QUALITY_REPORT_FILE,
    INITIAL_STATS_FILE,
    POSITIVE_SALES_FILE,
    RAW_DATA_FILE,
    STATISTICS_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    ensure_directories,
)


# =============================================================================
# 1. Missing Value Operations
# =============================================================================

def identify_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Identify missing values and calculate percentages for all columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input retail DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with Column, MissingCount, MissingPercentage, TotalRows,
        sorted by MissingCount descending.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows * 100).round(2)

    report = pd.DataFrame({
        "Column": df.columns,
        "MissingCount": missing_counts.values,
        "MissingPercentage": missing_pct.values,
        "TotalRows": total_rows,
    }).sort_values(by="MissingCount", ascending=False).reset_index(drop=True)

    return report


def investigate_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Investigate the nature of missing CustomerID and Description values.

    Analyzes whether missing CustomerID represents guest checkouts or data
    corruption, and whether missing Description correlates with administrative
    inventory write-offs (zero unit prices).

    Parameters
    ----------
    df : pd.DataFrame
        Input raw or intermediate DataFrame.

    Returns
    -------
    dict
        Detailed findings regarding missing records.
    """
    missing_desc = df[df["Description"].isnull()]
    missing_cust = df[df["CustomerID"].isnull()]

    # Check price distribution for missing Description
    desc_zero_price = (missing_desc["UnitPrice"] == 0).sum()
    desc_zero_price_pct = (
        round(desc_zero_price / len(missing_desc) * 100, 2)
        if len(missing_desc) > 0
        else 0.0
    )

    # Check transaction types for missing CustomerID
    cust_cancellations = missing_cust["InvoiceNo"].astype(str).str.startswith(CANCELLATION_PREFIX).sum()
    cust_total_amount = (missing_cust["Quantity"] * missing_cust["UnitPrice"]).sum()

    findings = {
        "missing_description_count": len(missing_desc),
        "missing_description_zero_price_count": int(desc_zero_price),
        "missing_description_zero_price_pct": desc_zero_price_pct,
        "missing_customer_count": len(missing_cust),
        "missing_customer_pct": round(len(missing_cust) / len(df) * 100, 2),
        "missing_customer_cancellations": int(cust_cancellations),
        "missing_customer_gross_amount": round(float(cust_total_amount), 2),
    }

    return findings


# =============================================================================
# 2. Duplicate Records Operations
# =============================================================================

def identify_duplicates(df: pd.DataFrame) -> Tuple[int, float, pd.DataFrame]:
    """Identify exact duplicate rows in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple of (int, float, pd.DataFrame)
        Duplicate row count, duplicate percentage, and sample duplicate rows.
    """
    dup_mask = df.duplicated(keep=False)
    dup_count = df.duplicated(keep="first").sum()
    dup_pct = round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0.0
    sample_dups = df[dup_mask].head(10)

    return int(dup_count), dup_pct, sample_dups


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove justified exact duplicate rows from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        Deduplicated DataFrame and number of duplicate rows removed.
    """
    initial_count = len(df)
    df_dedup = df.drop_duplicates(keep="first").copy()
    removed_count = initial_count - len(df_dedup)

    print(f"[Data Cleaning] Removed {removed_count:,} exact duplicate rows.")
    print(f"  Rows before : {initial_count:,}")
    print(f"  Rows after  : {len(df_dedup):,}")

    return df_dedup, removed_count


# =============================================================================
# 3. Text Normalization and Cleaning
# =============================================================================

def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text whitespace and casing for string columns.

    - Strips leading and trailing whitespaces from Description, StockCode, Country, InvoiceNo.
    - Normalizes multiple internal whitespace characters to single spaces.
    - Converts product descriptions to uppercase for consistency.
    - Corrects known anomalous country values (e.g. 'Unspecified' retained or documented).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized text fields.
    """
    df = df.copy()

    # InvoiceNo
    if "InvoiceNo" in df.columns:
        df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

    # StockCode
    if "StockCode" in df.columns:
        df["StockCode"] = df["StockCode"].astype(str).str.strip()

    # Description
    if "Description" in df.columns:
        df["Description"] = (
            df["Description"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.upper()
        )
        # Restore actual NaNs where text was 'NAN'
        df.loc[df["Description"] == "NAN", "Description"] = np.nan

    # Country
    if "Country" in df.columns:
        df["Country"] = df["Country"].astype(str).str.strip()

    print("[Data Cleaning] Text fields normalized: whitespace stripped, casing standardized.")
    return df


# =============================================================================
# 4. Data Type Corrections
# =============================================================================

def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Correct column data types for accurate analysis.

    - InvoiceDate: Converted to pandas datetime64[ns].
    - Quantity: Converted to integer (int64).
    - UnitPrice: Converted to float (float64).
    - InvoiceNo: Kept as string (object/string) to preserve alphanumeric codes (e.g. 'C536379').
    - StockCode: Kept as string (object/string) to preserve letters (e.g. '85123A', 'POST').
    - CustomerID: Formatted as clean string without decimal (e.g. '17850', 'Unknown' for NaNs).
    - Country: Kept as string category.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected data types.
    """
    df = df.copy()

    # InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Quantity to integer
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(np.int64)

    # UnitPrice to float
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").astype(np.float64)

    # InvoiceNo to string
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    # StockCode to string
    df["StockCode"] = df["StockCode"].astype(str)

    # CustomerID formatting: retain string representation, handle missing as 'Unknown'
    # Format float IDs like 17850.0 -> '17850'
    def format_customer_id(val: Any) -> str:
        if pd.isna(val) or val is None or str(val).strip() in ("", "nan", "NaN"):
            return "Unknown"
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val).strip()

    df["CustomerID"] = df["CustomerID"].apply(format_customer_id)

    # Country to string
    df["Country"] = df["Country"].astype(str)

    print("[Data Cleaning] Data types corrected successfully.")
    return df


# =============================================================================
# 5. Non-Commercial / Invalid Price Filtering
# =============================================================================

def filter_invalid_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Filter out non-commercial records while preserving cancellations.

    Business Rules:
    1. UnitPrice <= 0 records are non-commercial:
       - 2 negative price records (-11,062.06) are administrative bad debt adjustments.
       - Zero-price records are damaged goods, stock adjustments, samples, or warehouse audits.
       Neither represent legitimate consumer sales transactions.
    2. Legitimate cancellations (InvoiceNo starting with 'C' and Quantity < 0) are PRESERVED
       when UnitPrice > 0, as they represent genuine customer refund transactions.
    3. Legitimate sales (Quantity > 0 and UnitPrice > 0) are PRESERVED.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        Cleaned DataFrame and count of invalid records removed.
    """
    initial_count = len(df)

    # Keep records where UnitPrice > 0
    valid_price_mask = df["UnitPrice"] > 0
    invalid_price_count = (~valid_price_mask).sum()

    df_filtered = df[valid_price_mask].copy()
    removed_count = initial_count - len(df_filtered)

    print(f"[Data Cleaning] Filtered {removed_count:,} non-commercial / invalid price records (UnitPrice <= 0).")
    print(f"  Rows before : {initial_count:,}")
    print(f"  Rows after  : {len(df_filtered):,}")

    return df_filtered, removed_count


# =============================================================================
# 6. Master Cleaning Pipeline
# =============================================================================

def clean_retail_data(
    df: pd.DataFrame,
    save_cleaned: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the end-to-end data cleaning workflow.

    Pipeline Steps:
    1. Identify and document initial missing values and duplicates.
    2. Remove exact duplicate rows.
    3. Normalize string fields (whitespace, casing).
    4. Filter non-commercial invalid price records (UnitPrice <= 0).
    5. Convert columns to appropriate data types.
    6. Generate separate positive-sales dataset for commercial modeling.
    7. Optionally save output datasets to data/processed/.

    Parameters
    ----------
    df : pd.DataFrame
        Raw UCI Online Retail dataset.
    save_cleaned : bool, default True
        Whether to save cleaned datasets to disk.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (cleaned_full_transactions_df, positive_sales_df)
    """
    ensure_directories()
    initial_rows = len(df)
    print("=" * 70)
    print("STARTING DATA CLEANING PIPELINE")
    print("=" * 70)
    print(f"Initial raw record count: {initial_rows:,}")

    # Step 1: Remove exact duplicates
    df_step1, dupes_removed = remove_duplicates(df)

    # Step 2: Normalize text fields
    df_step2 = clean_text_fields(df_step1)

    # Step 3: Filter non-commercial records (UnitPrice <= 0)
    df_step3, invalid_price_removed = filter_invalid_records(df_step2)

    # Step 4: Correct data types
    df_cleaned = correct_data_types(df_step3)

    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)

    # Step 5: Separate positive sales dataset
    # Positive sales: Quantity > 0, UnitPrice > 0, not cancellation
    is_cancel = df_cleaned["InvoiceNo"].str.startswith(CANCELLATION_PREFIX)
    df_positive_sales = df_cleaned[(df_cleaned["Quantity"] > 0) & (~is_cancel)].copy().reset_index(drop=True)

    print("\n[Data Cleaning Summary]")
    print(f"  Raw records                : {initial_rows:,}")
    print(f"  Duplicates removed         : {dupes_removed:,}")
    print(f"  Invalid price rows removed : {invalid_price_removed:,}")
    print(f"  Final cleaned transactions : {len(df_cleaned):,}")
    print(f"  Cancellations preserved    : {is_cancel.sum():,}")
    print(f"  Positive sales records     : {len(df_positive_sales):,}")

    if save_cleaned:
        df_cleaned.to_csv(CLEANED_DATA_FILE, index=False)
        print(f"  💾 Saved cleaned data to: {CLEANED_DATA_FILE}")
        df_positive_sales.to_csv(POSITIVE_SALES_FILE, index=False)
        print(f"  💾 Saved positive sales data to: {POSITIVE_SALES_FILE}")

    print("=" * 70)
    print("DATA CLEANING PIPELINE COMPLETE")
    print("=" * 70)

    return df_cleaned, df_positive_sales


# =============================================================================
# 7. Descriptive Statistics and Before/After Comparison
# =============================================================================

def compute_detailed_statistics(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute detailed descriptive statistics for numerical columns.

    Includes: Count, Mean, Median, Std, Min, 25%, 50%, 75%, Max.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : list of str, optional
        Columns to evaluate. Defaults to Quantity and UnitPrice (and TotalAmount if present).

    Returns
    -------
    pd.DataFrame
        Descriptive statistics table.
    """
    if numeric_cols is None:
        numeric_cols = ["Quantity", "UnitPrice"]
        if "TotalAmount" in df.columns:
            numeric_cols.append("TotalAmount")

    stats_list = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        stats_list.append({
            "Column": col,
            "Count": int(series.count()),
            "Mean": round(float(series.mean()), 4),
            "Median": round(float(series.median()), 4),
            "Std": round(float(series.std()), 4),
            "Min": round(float(series.min()), 4),
            "25%": round(float(series.quantile(0.25)), 4),
            "50%": round(float(series.quantile(0.50)), 4),
            "75%": round(float(series.quantile(0.75)), 4),
            "Max": round(float(series.max()), 4),
        })

    return pd.DataFrame(stats_list)


def generate_before_after_comparison(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate a comprehensive before-and-after comparison table.

    Compares:
    - Total record count
    - Column count
    - Exact duplicate count
    - Missing Description count
    - Missing CustomerID count
    - Zero or negative price count
    - Negative quantity count
    - Cancellation record count
    - Quantity Mean, Median, Min, Max
    - UnitPrice Mean, Median, Min, Max

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw dataset before cleaning.
    cleaned_df : pd.DataFrame
        Cleaned dataset after pipeline execution.
    output_path : str or Path, optional
        Destination CSV path.

    Returns
    -------
    pd.DataFrame
        Comparison summary table.
    """
    raw_cancels = raw_df["InvoiceNo"].astype(str).str.startswith(CANCELLATION_PREFIX).sum()
    clean_cancels = cleaned_df["InvoiceNo"].astype(str).str.startswith(CANCELLATION_PREFIX).sum()

    raw_neg_qty = (raw_df["Quantity"] < 0).sum()
    clean_neg_qty = (cleaned_df["Quantity"] < 0).sum()

    raw_zero_neg_price = (raw_df["UnitPrice"] <= 0).sum()
    clean_zero_neg_price = (cleaned_df["UnitPrice"] <= 0).sum()

    raw_dupes = raw_df.duplicated().sum()
    clean_dupes = cleaned_df.duplicated().sum()

    comparison_data = [
        {"Metric": "Total Records", "Before Cleaning": len(raw_df), "After Cleaning": len(cleaned_df), "Change": len(cleaned_df) - len(raw_df)},
        {"Metric": "Exact Duplicate Rows", "Before Cleaning": int(raw_dupes), "After Cleaning": int(clean_dupes), "Change": int(clean_dupes - raw_dupes)},
        {"Metric": "Missing Description", "Before Cleaning": int(raw_df["Description"].isnull().sum()), "After Cleaning": int(cleaned_df["Description"].isnull().sum()), "Change": int(cleaned_df["Description"].isnull().sum() - raw_df["Description"].isnull().sum())},
        {"Metric": "Missing / Unknown CustomerID", "Before Cleaning": int(raw_df["CustomerID"].isnull().sum()), "After Cleaning": int((cleaned_df["CustomerID"] == "Unknown").sum()), "Change": "Preserved as 'Unknown'"},
        {"Metric": "Zero / Negative UnitPrice", "Before Cleaning": int(raw_zero_neg_price), "After Cleaning": int(clean_zero_neg_price), "Change": int(clean_zero_neg_price - raw_zero_neg_price)},
        {"Metric": "Negative Quantities (Returns/Cancellations)", "Before Cleaning": int(raw_neg_qty), "After Cleaning": int(clean_neg_qty), "Change": int(clean_neg_qty - raw_neg_qty)},
        {"Metric": "Cancellation Invoices ('C')", "Before Cleaning": int(raw_cancels), "After Cleaning": int(clean_cancels), "Change": int(clean_cancels - raw_cancels)},
        {"Metric": "Quantity: Mean", "Before Cleaning": round(float(raw_df["Quantity"].mean()), 2), "After Cleaning": round(float(cleaned_df["Quantity"].mean()), 2), "Change": round(float(cleaned_df["Quantity"].mean() - raw_df["Quantity"].mean()), 2)},
        {"Metric": "Quantity: Median", "Before Cleaning": round(float(raw_df["Quantity"].median()), 2), "After Cleaning": round(float(cleaned_df["Quantity"].median()), 2), "Change": round(float(cleaned_df["Quantity"].median() - raw_df["Quantity"].median()), 2)},
        {"Metric": "UnitPrice: Mean (£)", "Before Cleaning": round(float(raw_df["UnitPrice"].mean()), 2), "After Cleaning": round(float(cleaned_df["UnitPrice"].mean()), 2), "Change": round(float(cleaned_df["UnitPrice"].mean() - raw_df["UnitPrice"].mean()), 2)},
        {"Metric": "UnitPrice: Median (£)", "Before Cleaning": round(float(raw_df["UnitPrice"].median()), 2), "After Cleaning": round(float(cleaned_df["UnitPrice"].median()), 2), "Change": round(float(cleaned_df["UnitPrice"].median() - raw_df["UnitPrice"].median()), 2)},
        {"Metric": "UnitPrice: Min (£)", "Before Cleaning": round(float(raw_df["UnitPrice"].min()), 2), "After Cleaning": round(float(cleaned_df["UnitPrice"].min()), 2), "Change": "Removed negative prices"},
    ]

    comp_df = pd.DataFrame(comparison_data)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(out, index=False)
        print(f"[Data Cleaning] Comparison report saved to: {out}")

    return comp_df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from src.data_loader import load_raw_data

    ensure_directories()
    raw = load_raw_data()

    # Save initial stats
    init_stats = compute_detailed_statistics(raw)
    init_stats.to_csv(INITIAL_STATS_FILE, index=False)
    print(f"Saved initial statistics to: {INITIAL_STATS_FILE}")

    # Clean dataset
    cleaned, positive_sales = clean_retail_data(raw, save_cleaned=True)

    # Save cleaned stats
    clean_stats = compute_detailed_statistics(cleaned)
    clean_stats.to_csv(CLEANED_STATS_FILE, index=False)
    print(f"Saved cleaned statistics to: {CLEANED_STATS_FILE}")

    # Generate and save comparison report
    comp = generate_before_after_comparison(raw, cleaned, output_path=COMPARISON_REPORT_FILE)
    print("\n" + comp.to_string(index=False))

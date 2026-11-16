"""
Data exploration module for Online Retail Data Cleaning & Feature Engineering project.

Provides functions for exploring the raw UCI Online Retail dataset, including
summary statistics, missing-value analysis, duplicate detection, date-range
inspection, anomaly analysis (negative quantities, zero/negative prices,
cancellations), and country distribution.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.config import (
    CANCELLATION_PREFIX,
    DATA_QUALITY_REPORT_FILE,
    RAW_DATA_FILE,
    ensure_directories,
)


# =============================================================================
# 1. Basic Info
# =============================================================================

def basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Print and return basic information about the DataFrame.

    Displays shape, data types, and memory usage to stdout and returns a
    summary dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.

    Returns
    -------
    dict
        Keys: ``rows``, ``columns``, ``dtypes`` (dict of column→dtype string),
        ``memory_usage_mb`` (float), ``column_names`` (list[str]).
    """
    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print("=" * 60)
    print("BASIC DATASET INFORMATION")
    print("=" * 60)
    print(f"Shape          : {rows:,} rows × {cols} columns")
    print(f"Memory usage   : {mem_mb:.2f} MB")
    print(f"\nColumn data types:")
    for col in df.columns:
        print(f"  {col:<20s} {str(df[col].dtype)}")
    print("=" * 60)

    return {
        "rows": rows,
        "columns": cols,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "memory_usage_mb": round(mem_mb, 2),
        "column_names": list(df.columns),
    }


# =============================================================================
# 2. Missing-Value Analysis
# =============================================================================

def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse missing values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``Column``, ``MissingCount``,
        ``MissingPercent``, and ``TotalRows`` — one row per column that has
        at least one missing value, sorted by ``MissingCount`` descending.
    """
    total = len(df)
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    result = pd.DataFrame({
        "Column": missing_counts.index,
        "MissingCount": missing_counts.values,
        "MissingPercent": (missing_counts.values / total * 100).round(2),
        "TotalRows": total,
    })
    return result.reset_index(drop=True)


# =============================================================================
# 3. Duplicate Analysis
# =============================================================================

def duplicate_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.

    Returns
    -------
    dict
        Keys: ``duplicate_count`` (int), ``duplicate_percent`` (float),
        ``sample_duplicates`` (pd.DataFrame — up to 5 duplicated rows).
    """
    dup_mask = df.duplicated(keep=False)
    dup_count = df.duplicated(keep="first").sum()
    dup_pct = round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0.0
    sample = df[dup_mask].head(5)

    return {
        "duplicate_count": int(dup_count),
        "duplicate_percent": dup_pct,
        "sample_duplicates": sample,
    }


# =============================================================================
# 4. Unique-Value Counts
# =============================================================================

def unique_value_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Return the number of unique values for every column.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.

    Returns
    -------
    dict
        Mapping of column name → unique count.
    """
    return {col: int(df[col].nunique()) for col in df.columns}


# =============================================================================
# 5. Date-Range Analysis
# =============================================================================

def date_range_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Determine the date range of the InvoiceDate column.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset (must contain ``InvoiceDate``).

    Returns
    -------
    dict
        Keys: ``min_date``, ``max_date`` (pd.Timestamp),
        ``span_days`` (int).
    """
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    min_date = dates.min()
    max_date = dates.max()
    span = (max_date - min_date).days

    return {
        "min_date": min_date,
        "max_date": max_date,
        "span_days": int(span),
    }


# =============================================================================
# 6. Negative-Quantity Analysis
# =============================================================================

def negative_quantity_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse rows where Quantity is negative.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.

    Returns
    -------
    dict
        Keys: ``count`` (int), ``percent`` (float),
        ``sample_rows`` (pd.DataFrame — up to 5 rows).
    """
    neg_mask = df["Quantity"] < 0
    count = int(neg_mask.sum())
    pct = round(count / len(df) * 100, 2) if len(df) > 0 else 0.0
    sample = df[neg_mask].head(5)

    return {
        "count": count,
        "percent": pct,
        "sample_rows": sample,
    }


# =============================================================================
# 7. Zero / Negative Price Analysis
# =============================================================================

def zero_negative_price_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse rows with zero or negative UnitPrice.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.

    Returns
    -------
    dict
        Keys: ``zero_price_count``, ``negative_price_count`` (int),
        ``zero_price_percent``, ``negative_price_percent`` (float),
        ``sample_zero`` (pd.DataFrame — up to 5 rows),
        ``sample_negative`` (pd.DataFrame — up to 5 rows).
    """
    total = len(df)
    zero_mask = df["UnitPrice"] == 0
    neg_mask = df["UnitPrice"] < 0

    return {
        "zero_price_count": int(zero_mask.sum()),
        "negative_price_count": int(neg_mask.sum()),
        "zero_price_percent": round(zero_mask.sum() / total * 100, 2) if total else 0.0,
        "negative_price_percent": round(neg_mask.sum() / total * 100, 2) if total else 0.0,
        "sample_zero": df[zero_mask].head(5),
        "sample_negative": df[neg_mask].head(5),
    }


# =============================================================================
# 8. Cancellation Analysis
# =============================================================================

def cancellation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse cancelled invoices (InvoiceNo starting with 'C').

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.

    Returns
    -------
    dict
        Keys: ``cancellation_count`` (int), ``cancellation_percent`` (float),
        ``sample_cancellations`` (pd.DataFrame — up to 5 rows).
    """
    cancel_mask = df["InvoiceNo"].astype(str).str.startswith(CANCELLATION_PREFIX)
    count = int(cancel_mask.sum())
    pct = round(count / len(df) * 100, 2) if len(df) > 0 else 0.0

    return {
        "cancellation_count": count,
        "cancellation_percent": pct,
        "sample_cancellations": df[cancel_mask].head(5),
    }


# =============================================================================
# 9. Country Analysis
# =============================================================================

def country_analysis(df: pd.DataFrame) -> pd.Series:
    """Return value counts of the Country column.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.

    Returns
    -------
    pd.Series
        Counts sorted descending.
    """
    return df["Country"].value_counts()


# =============================================================================
# 10. Generate Data Quality Report
# =============================================================================

def generate_data_quality_report(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a comprehensive text-based data-quality report.

    Calls every exploration function above and assembles the results into a
    single human-readable string.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Online Retail dataset.
    output_path : str or Path, optional
        If provided, the report is saved to this file (parent dirs created
        automatically).

    Returns
    -------
    str
        The full report as a string.
    """
    lines: List[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    # --- Basic info --------------------------------------------------------
    info = basic_info(df)
    section("1. BASIC DATASET INFORMATION")
    lines.append(f"Rows           : {info['rows']:,}")
    lines.append(f"Columns        : {info['columns']}")
    lines.append(f"Memory usage   : {info['memory_usage_mb']} MB")
    lines.append(f"Column names   : {', '.join(info['column_names'])}")
    lines.append("\nData types:")
    for col, dtype in info["dtypes"].items():
        lines.append(f"  {col:<20s} {dtype}")

    # --- Missing values ----------------------------------------------------
    mv = missing_value_analysis(df)
    section("2. MISSING VALUE ANALYSIS")
    if mv.empty:
        lines.append("No missing values found.")
    else:
        lines.append(mv.to_string(index=False))

    # --- Duplicates --------------------------------------------------------
    dups = duplicate_analysis(df)
    section("3. DUPLICATE ANALYSIS")
    lines.append(f"Duplicate rows : {dups['duplicate_count']:,}")
    lines.append(f"Percentage     : {dups['duplicate_percent']}%")
    if not dups["sample_duplicates"].empty:
        lines.append("\nSample duplicate rows:")
        lines.append(dups["sample_duplicates"].to_string(index=False))

    # --- Unique values -----------------------------------------------------
    uv = unique_value_counts(df)
    section("4. UNIQUE VALUE COUNTS")
    for col, cnt in uv.items():
        lines.append(f"  {col:<20s} {cnt:,}")

    # --- Date range --------------------------------------------------------
    dr = date_range_analysis(df)
    section("5. DATE RANGE ANALYSIS")
    lines.append(f"Min date       : {dr['min_date']}")
    lines.append(f"Max date       : {dr['max_date']}")
    lines.append(f"Span (days)    : {dr['span_days']:,}")

    # --- Negative quantities -----------------------------------------------
    nq = negative_quantity_analysis(df)
    section("6. NEGATIVE QUANTITY ANALYSIS")
    lines.append(f"Count          : {nq['count']:,}")
    lines.append(f"Percentage     : {nq['percent']}%")
    if not nq["sample_rows"].empty:
        lines.append("\nSample rows with negative quantity:")
        lines.append(nq["sample_rows"].to_string(index=False))

    # --- Zero / negative prices -------------------------------------------
    zp = zero_negative_price_analysis(df)
    section("7. ZERO / NEGATIVE PRICE ANALYSIS")
    lines.append(f"Zero prices    : {zp['zero_price_count']:,}  ({zp['zero_price_percent']}%)")
    lines.append(f"Negative prices: {zp['negative_price_count']:,}  ({zp['negative_price_percent']}%)")
    if not zp["sample_zero"].empty:
        lines.append("\nSample rows with zero price:")
        lines.append(zp["sample_zero"].to_string(index=False))
    if not zp["sample_negative"].empty:
        lines.append("\nSample rows with negative price:")
        lines.append(zp["sample_negative"].to_string(index=False))

    # --- Cancellations -----------------------------------------------------
    ca = cancellation_analysis(df)
    section("8. CANCELLATION ANALYSIS")
    lines.append(f"Cancellations  : {ca['cancellation_count']:,}")
    lines.append(f"Percentage     : {ca['cancellation_percent']}%")
    if not ca["sample_cancellations"].empty:
        lines.append("\nSample cancellation rows:")
        lines.append(ca["sample_cancellations"].to_string(index=False))

    # --- Country distribution ----------------------------------------------
    ctry = country_analysis(df)
    section("9. COUNTRY DISTRIBUTION")
    lines.append(f"Number of countries: {len(ctry)}")
    lines.append("")
    lines.append(ctry.to_string())

    # --- Footer ------------------------------------------------------------
    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF DATA QUALITY REPORT")
    lines.append("=" * 70)

    report = "\n".join(lines)

    # Save if requested
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nReport saved to {out}")

    return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from src.data_loader import load_raw_data

    ensure_directories()

    print("Loading dataset …")
    df = load_raw_data()

    if df is not None:
        report = generate_data_quality_report(df, output_path=DATA_QUALITY_REPORT_FILE)
        print(report)
    else:
        print("Failed to load dataset. Check that the raw data file exists.")

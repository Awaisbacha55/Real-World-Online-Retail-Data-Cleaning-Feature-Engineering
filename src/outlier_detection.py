"""
Outlier detection module for Online Retail Data Cleaning & Feature Engineering project.

Provides IQR-based and Z-score-based outlier detection, flagging, capping
(winsorization), and summary reporting for numeric columns in the UCI
Online Retail dataset.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import IQR_MULTIPLIER, ZSCORE_THRESHOLD


# =============================================================================
# Core Detection Functions
# =============================================================================


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = IQR_MULTIPLIER,
) -> Dict[str, object]:
    """Detect outliers in a numeric series using the IQR method.

    Parameters
    ----------
    series : pd.Series
        Numeric series to analyse.
    multiplier : float, optional
        IQR multiplier for determining fence boundaries.
        Defaults to ``IQR_MULTIPLIER`` from config (1.5).

    Returns
    -------
    dict
        Keys:
        - q1, q3, iqr          – quartile statistics
        - lower_bound, upper_bound – fence boundaries
        - outlier_count         – number of outliers
        - outlier_percentage    – percentage of total values
        - outlier_indices       – pd.Index of outlier row labels
    """
    clean = series.dropna()

    q1: float = clean.quantile(0.25)
    q3: float = clean.quantile(0.75)
    iqr: float = q3 - q1

    lower_bound: float = q1 - multiplier * iqr
    upper_bound: float = q3 + multiplier * iqr

    outlier_mask = (clean < lower_bound) | (clean > upper_bound)
    outlier_indices = clean[outlier_mask].index

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(outlier_mask.sum()),
        "outlier_percentage": round(outlier_mask.sum() / len(clean) * 100, 2) if len(clean) else 0.0,
        "outlier_indices": outlier_indices,
    }


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = ZSCORE_THRESHOLD,
) -> Dict[str, object]:
    """Detect outliers in a numeric series using the Z-score method.

    Parameters
    ----------
    series : pd.Series
        Numeric series to analyse.
    threshold : float, optional
        Absolute Z-score value above which a point is considered an outlier.
        Defaults to ``ZSCORE_THRESHOLD`` from config (3.0).

    Returns
    -------
    dict
        Keys:
        - threshold             – threshold used
        - mean, std             – descriptive statistics
        - outlier_count         – number of outliers
        - outlier_percentage    – percentage of total values
        - outlier_indices       – pd.Index of outlier row labels
    """
    clean = series.dropna()
    mean: float = clean.mean()
    std: float = clean.std()

    if std == 0:
        # No variation → no outliers
        return {
            "threshold": threshold,
            "mean": mean,
            "std": std,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "outlier_indices": pd.Index([]),
        }

    z_scores = (clean - mean) / std
    outlier_mask = z_scores.abs() > threshold
    outlier_indices = clean[outlier_mask].index

    return {
        "threshold": threshold,
        "mean": mean,
        "std": std,
        "outlier_count": int(outlier_mask.sum()),
        "outlier_percentage": round(outlier_mask.sum() / len(clean) * 100, 2) if len(clean) else 0.0,
        "outlier_indices": outlier_indices,
    }


# =============================================================================
# Analysis & Reporting
# =============================================================================


def analyze_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Run both IQR and Z-score outlier detection on selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (cleaned retail data).
    columns : list of str, optional
        Columns to analyse.  Defaults to ``['Quantity', 'UnitPrice']``;
        ``'TotalAmount'`` is appended automatically when present in *df*.

    Returns
    -------
    dict
        Nested dict keyed by column name → method name ('iqr' / 'zscore') →
        detection result dict.
    """
    if columns is None:
        columns = ["Quantity", "UnitPrice"]
        if "TotalAmount" in df.columns and "TotalAmount" not in columns:
            columns.append("TotalAmount")

    results: Dict[str, Dict[str, object]] = {}
    for col in columns:
        if col not in df.columns:
            print(f"  ⚠ Column '{col}' not found in DataFrame — skipping.")
            continue
        results[col] = {
            "iqr": detect_outliers_iqr(df[col]),
            "zscore": detect_outliers_zscore(df[col]),
        }

    return results


def get_outlier_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a summary table of outlier statistics for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Columns to summarise (same defaulting logic as
        :func:`analyze_outliers`).

    Returns
    -------
    pd.DataFrame
        Summary with columns: Column, IQR_Outliers, IQR_Pct,
        ZScore_Outliers, ZScore_Pct, Min, Max, Mean, Median.
    """
    analysis = analyze_outliers(df, columns)

    rows: List[Dict[str, object]] = []
    for col, methods in analysis.items():
        series = df[col].dropna()
        rows.append(
            {
                "Column": col,
                "IQR_Outliers": methods["iqr"]["outlier_count"],
                "IQR_Pct": methods["iqr"]["outlier_percentage"],
                "ZScore_Outliers": methods["zscore"]["outlier_count"],
                "ZScore_Pct": methods["zscore"]["outlier_percentage"],
                "Min": series.min(),
                "Max": series.max(),
                "Mean": round(series.mean(), 2),
                "Median": series.median(),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Flagging & Capping
# =============================================================================


def flag_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    multiplier: float = IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Add boolean outlier-flag columns to *df* (non-destructive).

    For each column a new column ``<Column>_outlier`` is added with
    ``True`` where the value is an outlier.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Columns to flag (same defaulting logic as :func:`analyze_outliers`).
    method : {'iqr', 'zscore'}, optional
        Detection method to use.  Defaults to ``'iqr'``.
    multiplier : float, optional
        IQR multiplier (used only when *method* is ``'iqr'``).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with new boolean ``<Column>_outlier`` columns.
    """
    df = df.copy()

    if columns is None:
        columns = ["Quantity", "UnitPrice"]
        if "TotalAmount" in df.columns and "TotalAmount" not in columns:
            columns.append("TotalAmount")

    for col in columns:
        if col not in df.columns:
            print(f"  ⚠ Column '{col}' not found — skipping flag.")
            continue

        if method == "iqr":
            result = detect_outliers_iqr(df[col], multiplier=multiplier)
        elif method == "zscore":
            result = detect_outliers_zscore(df[col])
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

        df[f"{col}_outlier"] = df.index.isin(result["outlier_indices"])

    return df


def cap_outliers(
    df: pd.DataFrame,
    column: str,
    lower_percentile: float = 1,
    upper_percentile: float = 99,
) -> pd.DataFrame:
    """Cap (winsorise) outlier values at the given percentiles.

    Values below the *lower_percentile* are set to that percentile's value;
    values above the *upper_percentile* are set to that percentile's value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to winsorise.
    lower_percentile : float, optional
        Lower capping percentile (default 1).
    upper_percentile : float, optional
        Upper capping percentile (default 99).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the column values capped.
    """
    df = df.copy()

    lower_val = np.percentile(df[column].dropna(), lower_percentile)
    upper_val = np.percentile(df[column].dropna(), upper_percentile)

    before_min = df[column].min()
    before_max = df[column].max()

    df[column] = df[column].clip(lower=lower_val, upper=upper_val)

    print(f"  Capped '{column}': [{before_min}, {before_max}] → [{lower_val:.2f}, {upper_val:.2f}]")

    return df


# =============================================================================
# Main Demo
# =============================================================================


if __name__ == "__main__":
    from src.config import FEATURE_ENGINEERED_FILE, CLEANED_DATA_FILE
    from src.data_loader import load_raw_data

    print("=" * 70)
    print("  OUTLIER DETECTION — Online Retail Dataset")
    print("=" * 70)

    if FEATURE_ENGINEERED_FILE.exists():
        print(f"Loading data from {FEATURE_ENGINEERED_FILE} …")
        df = pd.read_csv(FEATURE_ENGINEERED_FILE)
    elif CLEANED_DATA_FILE.exists():
        print(f"Loading data from {CLEANED_DATA_FILE} …")
        df = pd.read_csv(CLEANED_DATA_FILE)
    else:
        df = load_raw_data()

    if df is None:
        raise SystemExit("Failed to load data.")

    # Compute TotalAmount if not present
    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # --- Full analysis -------------------------------------------------------
    print("\n📊 Outlier Analysis (IQR & Z-Score)")
    print("-" * 50)
    analysis = analyze_outliers(df)
    for col, methods in analysis.items():
        iqr_res = methods["iqr"]
        z_res = methods["zscore"]
        print(f"\n  Column: {col}")
        print(f"    IQR  → bounds [{iqr_res['lower_bound']:.2f}, {iqr_res['upper_bound']:.2f}] | "
              f"outliers: {iqr_res['outlier_count']:,} ({iqr_res['outlier_percentage']:.2f}%)")
        print(f"    Z    → threshold {z_res['threshold']} | "
              f"outliers: {z_res['outlier_count']:,} ({z_res['outlier_percentage']:.2f}%)")

    # --- Summary table -------------------------------------------------------
    print("\n📋 Outlier Summary Table")
    print("-" * 50)
    summary = get_outlier_summary(df)
    print(summary.to_string(index=False))

    # --- Flag outliers -------------------------------------------------------
    print("\n🏷️  Flagging outliers (IQR method) …")
    df_flagged = flag_outliers(df)
    flag_cols = [c for c in df_flagged.columns if c.endswith("_outlier")]
    for fc in flag_cols:
        n = df_flagged[fc].sum()
        print(f"    {fc}: {n:,} flagged")

    # --- Capping demo --------------------------------------------------------
    print("\n✂️  Capping demo (Quantity, 1st–99th percentile) …")
    df_capped = cap_outliers(df.copy(), "Quantity")

    print("\n✅ Outlier detection module executed successfully.")

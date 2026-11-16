"""
Feature Engineering module for Online Retail Data Cleaning & Feature Engineering project.

Creates new features from the cleaned UCI Online Retail dataset, including
transaction-level features (totals, date parts, flags, categories) and
aggregation-based features (customer frequency, invoice-level stats).

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

import numpy as np
import pandas as pd

from src.config import CANCELLATION_PREFIX


# =============================================================================
# Quantity & Price Bin Definitions
# =============================================================================

QUANTITY_BINS = [0, 2, 12, 50, np.inf]
QUANTITY_LABELS = ["Low", "Medium", "High", "Very High"]

PRICE_BINS = [0, 2.0, 10.0, 50.0, np.inf]
PRICE_LABELS = ["Budget", "Standard", "Premium", "Luxury"]

# =============================================================================
# Feature Names (single source of truth)
# =============================================================================

TRANSACTION_FEATURES = [
    "TotalAmount",
    "InvoiceYear",
    "InvoiceMonth",
    "InvoiceDay",
    "InvoiceHour",
    "PurchaseDayOfWeek",
    "IsCancellation",
    "IsReturn",
    "IsWeekend",
    "QuantityCategory",
    "PriceCategory",
]

AGGREGATION_FEATURES = [
    "CustomerPurchaseFrequency",
    "InvoiceItemCount",
    "AverageInvoiceValue",
]

ALL_FEATURES = TRANSACTION_FEATURES + AGGREGATION_FEATURES


# =============================================================================
# Transaction-Level Features
# =============================================================================


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transaction-level features from the cleaned dataset.

    Features created:
        TotalAmount         – Quantity * UnitPrice
        InvoiceYear         – Year extracted from InvoiceDate
        InvoiceMonth        – Month extracted from InvoiceDate
        InvoiceDay          – Day extracted from InvoiceDate
        InvoiceHour         – Hour extracted from InvoiceDate
        PurchaseDayOfWeek   – Weekday (0=Monday … 6=Sunday)
        IsCancellation      – True when InvoiceNo starts with the cancellation prefix
        IsReturn            – True when Quantity < 0
        IsWeekend           – True when PurchaseDayOfWeek >= 5
        QuantityCategory    – Ordinal bucket for absolute Quantity
        PriceCategory       – Ordinal bucket for UnitPrice

    Args:
        df: Cleaned retail DataFrame with at least the columns
            InvoiceNo, Quantity, InvoiceDate, UnitPrice.

    Returns:
        DataFrame with the original columns plus the new transaction features.
    """
    df = df.copy()

    # --- Monetary ---
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # --- Date / time parts ---
    invoice_dt = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceYear"] = invoice_dt.dt.year
    df["InvoiceMonth"] = invoice_dt.dt.month
    df["InvoiceDay"] = invoice_dt.dt.day
    df["InvoiceHour"] = invoice_dt.dt.hour
    df["PurchaseDayOfWeek"] = invoice_dt.dt.day_of_week  # 0=Mon … 6=Sun

    # --- Boolean flags ---
    df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith(
        CANCELLATION_PREFIX
    )
    df["IsReturn"] = df["Quantity"] < 0
    df["IsWeekend"] = df["PurchaseDayOfWeek"] >= 5

    # --- Categorical bins ---
    abs_qty = df["Quantity"].abs()
    df["QuantityCategory"] = pd.cut(
        abs_qty,
        bins=QUANTITY_BINS,
        labels=QUANTITY_LABELS,
        include_lowest=True,
        right=True,
    )

    df["PriceCategory"] = pd.cut(
        df["UnitPrice"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        include_lowest=True,
        right=True,
    )

    print(f"[Feature Engineering] Created {len(TRANSACTION_FEATURES)} transaction features.")
    return df


# =============================================================================
# Aggregation-Level Features
# =============================================================================


def create_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregation-based features and merge them back to the row level.

    Features created:
        CustomerPurchaseFrequency – Number of unique invoices per CustomerID
        InvoiceItemCount          – Number of line items per InvoiceNo
        AverageInvoiceValue       – Mean TotalAmount per InvoiceNo

    TotalAmount must already exist in *df* (call
    ``create_transaction_features`` first, or ensure the column is present).

    Args:
        df: DataFrame that already contains TotalAmount, InvoiceNo,
            and CustomerID columns.

    Returns:
        DataFrame with the new aggregation features merged in.
    """
    df = df.copy()

    # Ensure TotalAmount exists
    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # --- Customer Purchase Frequency ---
    if "CustomerID" in df.columns:
        customer_freq = (
            df.dropna(subset=["CustomerID"])
            .groupby("CustomerID")["InvoiceNo"]
            .nunique()
            .rename("CustomerPurchaseFrequency")
        )
        df = df.merge(customer_freq, on="CustomerID", how="left")
    else:
        df["CustomerPurchaseFrequency"] = np.nan

    # --- Invoice Item Count ---
    invoice_item_count = (
        df.groupby("InvoiceNo")
        .size()
        .rename("InvoiceItemCount")
    )
    df = df.merge(invoice_item_count, on="InvoiceNo", how="left")

    # --- Average Invoice Value ---
    avg_invoice_value = (
        df.groupby("InvoiceNo")["TotalAmount"]
        .mean()
        .rename("AverageInvoiceValue")
    )
    df = df.merge(avg_invoice_value, on="InvoiceNo", how="left")

    print(f"[Feature Engineering] Created {len(AGGREGATION_FEATURES)} aggregation features.")
    return df


# =============================================================================
# Master Function
# =============================================================================


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    Calls :func:`create_transaction_features` followed by
    :func:`create_aggregation_features`.

    Args:
        df: Cleaned retail DataFrame.

    Returns:
        DataFrame with all engineered features attached.
    """
    print("[Feature Engineering] Starting feature engineering pipeline …")

    df = create_transaction_features(df)
    df = create_aggregation_features(df)

    print(
        f"[Feature Engineering] Pipeline complete. "
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns."
    )
    return df


# =============================================================================
# Validation
# =============================================================================


def validate_features(df: pd.DataFrame) -> dict:
    """Validate that all engineered features exist and have sensible values.

    Checks performed per feature:
        * Existence in the DataFrame
        * Non-null ratio
        * Value-range / category checks where applicable

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Dictionary with keys ``'passed'`` (bool), ``'checks'`` (list of
        per-feature dicts), and ``'summary'`` (str).
    """
    checks: list[dict] = []
    all_passed = True

    for feature in ALL_FEATURES:
        entry: dict = {"feature": feature, "exists": feature in df.columns}
        if not entry["exists"]:
            entry["passed"] = False
            all_passed = False
            checks.append(entry)
            continue

        series = df[feature]
        non_null = series.notna().sum()
        total = len(series)
        entry["non_null_count"] = int(non_null)
        entry["non_null_pct"] = round(non_null / total * 100, 2) if total else 0.0

        # Feature-specific range checks
        passed = True
        if feature == "InvoiceYear":
            if series.dropna().min() < 2000 or series.dropna().max() > 2030:
                passed = False
                entry["issue"] = "Year values outside 2000-2030 range"
        elif feature == "InvoiceMonth":
            if series.dropna().min() < 1 or series.dropna().max() > 12:
                passed = False
                entry["issue"] = "Month values outside 1-12 range"
        elif feature == "InvoiceDay":
            if series.dropna().min() < 1 or series.dropna().max() > 31:
                passed = False
                entry["issue"] = "Day values outside 1-31 range"
        elif feature == "InvoiceHour":
            if series.dropna().min() < 0 or series.dropna().max() > 23:
                passed = False
                entry["issue"] = "Hour values outside 0-23 range"
        elif feature == "PurchaseDayOfWeek":
            if series.dropna().min() < 0 or series.dropna().max() > 6:
                passed = False
                entry["issue"] = "Day-of-week values outside 0-6 range"
        elif feature in ("IsCancellation", "IsReturn", "IsWeekend"):
            unique = set(series.dropna().unique())
            if not unique.issubset({True, False}):
                passed = False
                entry["issue"] = f"Expected boolean values, got {unique}"
        elif feature == "QuantityCategory":
            valid = set(QUANTITY_LABELS)
            actual = set(series.dropna().unique())
            if not actual.issubset(valid):
                passed = False
                entry["issue"] = f"Unexpected categories: {actual - valid}"
        elif feature == "PriceCategory":
            valid = set(PRICE_LABELS)
            actual = set(series.dropna().unique())
            if not actual.issubset(valid):
                passed = False
                entry["issue"] = f"Unexpected categories: {actual - valid}"
        elif feature in ("CustomerPurchaseFrequency", "InvoiceItemCount"):
            if series.dropna().min() < 0:
                passed = False
                entry["issue"] = "Negative count detected"

        entry["passed"] = passed
        if not passed:
            all_passed = False
        checks.append(entry)

    n_passed = sum(1 for c in checks if c["passed"])
    summary = (
        f"Validation {'PASSED' if all_passed else 'FAILED'}: "
        f"{n_passed}/{len(checks)} feature checks passed."
    )
    print(f"[Feature Engineering] {summary}")

    return {"passed": all_passed, "checks": checks, "summary": summary}


# =============================================================================
# Feature Descriptions
# =============================================================================


def get_feature_descriptions() -> dict[str, str]:
    """Return a mapping of engineered feature names to human-readable descriptions.

    Returns:
        Dictionary ``{feature_name: description}``.
    """
    return {
        "TotalAmount": "Line-item revenue: Quantity × UnitPrice.",
        "InvoiceYear": "Year extracted from InvoiceDate.",
        "InvoiceMonth": "Month (1-12) extracted from InvoiceDate.",
        "InvoiceDay": "Day of the month (1-31) extracted from InvoiceDate.",
        "InvoiceHour": "Hour of the day (0-23) extracted from InvoiceDate.",
        "PurchaseDayOfWeek": "Day of the week (0=Monday … 6=Sunday) from InvoiceDate.",
        "IsCancellation": "True if InvoiceNo starts with the cancellation prefix ('C').",
        "IsReturn": "True if Quantity is negative (goods returned).",
        "IsWeekend": "True if the purchase occurred on Saturday or Sunday.",
        "QuantityCategory": (
            "Ordinal category for absolute Quantity: "
            "Low (0-2), Medium (3-12), High (13-50), Very High (>50)."
        ),
        "PriceCategory": (
            "Ordinal category for UnitPrice: "
            "Budget (0-2), Standard (2-10), Premium (10-50), Luxury (>50)."
        ),
        "CustomerPurchaseFrequency": (
            "Number of unique invoices per CustomerID (purchase frequency)."
        ),
        "InvoiceItemCount": "Number of line items (rows) in each invoice.",
        "AverageInvoiceValue": "Mean TotalAmount across all line items of an invoice.",
    }


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    from src.config import CLEANED_DATA_FILE, FEATURE_ENGINEERED_FILE, ensure_directories

    ensure_directories()

    # Load cleaned data
    if CLEANED_DATA_FILE.exists():
        print(f"Loading cleaned data from {CLEANED_DATA_FILE} …")
        df = pd.read_csv(CLEANED_DATA_FILE, parse_dates=["InvoiceDate"])
    else:
        print(
            f"Cleaned data not found at {CLEANED_DATA_FILE}. "
            "Run data_cleaning.py first."
        )
        raise SystemExit(1)

    # Engineer features
    df = engineer_all_features(df)

    # Validate
    report = validate_features(df)
    print()
    for check in report["checks"]:
        status = "✓" if check["passed"] else "✗"
        extra = f" — {check.get('issue', '')}" if not check["passed"] else ""
        print(f"  {status} {check['feature']}{extra}")
    print()
    print(report["summary"])

    # Save
    df.to_csv(FEATURE_ENGINEERED_FILE, index=False)
    print(f"\nSaved feature-engineered data to {FEATURE_ENGINEERED_FILE}")

    # Show descriptions
    print("\n— Feature Descriptions —")
    for name, desc in get_feature_descriptions().items():
        print(f"  {name}: {desc}")

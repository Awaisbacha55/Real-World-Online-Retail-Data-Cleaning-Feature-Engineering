"""
Unit tests for data cleaning operations.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Domain: Data Science
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

import numpy as np
import pandas as pd
import pytest

from src.config import CANCELLATION_PREFIX, ORIGINAL_COLUMNS, RAW_DATA_FILE
from src.data_cleaning import (
    clean_retail_data,
    clean_text_fields,
    compute_detailed_statistics,
    correct_data_types,
    filter_invalid_records,
    identify_duplicates,
    identify_missing_values,
    investigate_missing_data,
    remove_duplicates,
)
from src.data_loader import load_raw_data


@pytest.fixture(scope="session")
def sample_raw_data():
    """Create a controlled sample dataset mimicking the UCI Online Retail dataset."""
    data = {
        "InvoiceNo": ["536365", "536365", "C536379", "536381", "536382", "536382", "A563185"],
        "StockCode": ["85123A ", " 71053", "D", "84406B", "22197", "22197", "B"],
        "Description": [" white hanging heart ", "WHITE METAL LANTERN", "Discount", "CREAM CUPID", None, None, "Adjust bad debt"],
        "Quantity": [6, 6, -1, 4, 10, 10, 1],
        "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 08:26:00", "2010-12-01 09:41:00", "2010-12-01 10:11:00", "2010-12-01 11:22:00", "2010-12-01 11:22:00", "2011-08-12 14:50:00"],
        "UnitPrice": [2.55, 3.39, 27.50, 2.75, 0.00, 0.00, -11062.06],
        "CustomerID": [17850.0, 17850.0, 14527.0, np.nan, 15311.0, 15311.0, np.nan],
        "Country": ["United Kingdom ", "United Kingdom", "United Kingdom", "France", "Germany", "Germany", "United Kingdom"],
    }
    return pd.DataFrame(data)


def test_dataset_loading():
    """Verify that the dataset loader successfully loads records and checks structure."""
    df = load_raw_data()
    assert df is not None
    assert len(df) > 500000
    for col in ORIGINAL_COLUMNS:
        assert col in df.columns, f"Expected column {col} missing from loaded data"


def test_identify_missing_values(sample_raw_data):
    """Test missing value identification and percentage calculations."""
    missing_report = identify_missing_values(sample_raw_data)
    assert isinstance(missing_report, pd.DataFrame)
    assert "MissingCount" in missing_report.columns
    assert "MissingPercentage" in missing_report.columns

    desc_missing = missing_report.loc[missing_report["Column"] == "Description", "MissingCount"].values[0]
    assert desc_missing == 2


def test_investigate_missing_data(sample_raw_data):
    """Test missing value investigation for Description and CustomerID."""
    findings = investigate_missing_data(sample_raw_data)
    assert findings["missing_description_count"] == 2
    assert findings["missing_customer_count"] == 2
    # In sample data, all missing descriptions have UnitPrice == 0
    assert findings["missing_description_zero_price_count"] == 2


def test_duplicate_identification_and_removal(sample_raw_data):
    """Verify exact duplicate identification and removal."""
    # Row 4 and 5 are exact duplicates
    dup_count, dup_pct, sample_dups = identify_duplicates(sample_raw_data)
    assert dup_count == 1
    assert dup_pct > 0

    df_dedup, removed = remove_duplicates(sample_raw_data)
    assert removed == 1
    assert len(df_dedup) == len(sample_raw_data) - 1
    assert df_dedup.duplicated().sum() == 0


def test_text_normalization(sample_raw_data):
    """Test stripping whitespace and casing normalization."""
    cleaned = clean_text_fields(sample_raw_data)
    # Check that leading/trailing spaces are removed
    assert cleaned.loc[0, "StockCode"] == "85123A"
    assert cleaned.loc[1, "StockCode"] == "71053"
    assert cleaned.loc[0, "Country"] == "United Kingdom"
    assert cleaned.loc[0, "Description"] == "WHITE HANGING HEART"


def test_non_commercial_filtering(sample_raw_data):
    """Test filtering of non-commercial records (UnitPrice <= 0) while preserving cancellations."""
    filtered, removed = filter_invalid_records(sample_raw_data)
    # 2 rows with 0.00 and 1 row with -11062.06 should be removed
    assert (filtered["UnitPrice"] <= 0).sum() == 0
    assert removed == 3
    # Cancellation invoice C536379 has UnitPrice 27.50, so it must be preserved!
    assert (filtered["InvoiceNo"] == "C536379").sum() == 1


def test_data_type_corrections(sample_raw_data):
    """Test conversion of columns to appropriate target data types."""
    converted = correct_data_types(sample_raw_data)
    assert pd.api.types.is_datetime64_any_dtype(converted["InvoiceDate"])
    assert pd.api.types.is_integer_dtype(converted["Quantity"])
    assert pd.api.types.is_float_dtype(converted["UnitPrice"])
    assert converted.loc[0, "CustomerID"] == "17850"
    assert converted.loc[3, "CustomerID"] == "Unknown"


def test_clean_retail_data_end_to_end(sample_raw_data):
    """Test full cleaning pipeline and positive sales separation."""
    df_cleaned, df_pos_sales = clean_retail_data(sample_raw_data, save_cleaned=False)

    # Cleaned transactions: 7 raw rows - 1 duplicate = 6 rows.
    # From 6 rows, 2 invalid price rows (0.00 and -11062.06) removed -> 4 rows.
    assert len(df_cleaned) == 4
    assert (df_cleaned["UnitPrice"] <= 0).sum() == 0

    # Cancellations should be preserved in df_cleaned (row 2: C536379)
    assert (df_cleaned["InvoiceNo"].str.startswith(CANCELLATION_PREFIX)).sum() == 1

    # Positive sales contains only Quantity > 0 and no cancellations (3 rows)
    assert len(df_pos_sales) == 3
    assert (df_pos_sales["Quantity"] <= 0).sum() == 0
    assert (df_pos_sales["InvoiceNo"].str.startswith(CANCELLATION_PREFIX)).sum() == 0


def test_descriptive_statistics_computation(sample_raw_data):
    """Test calculation of descriptive statistics before and after cleaning."""
    stats = compute_detailed_statistics(sample_raw_data)
    assert isinstance(stats, pd.DataFrame)
    assert "Mean" in stats.columns
    assert "Median" in stats.columns
    assert "Std" in stats.columns
    assert "Min" in stats.columns
    assert "Max" in stats.columns
    assert len(stats) >= 2

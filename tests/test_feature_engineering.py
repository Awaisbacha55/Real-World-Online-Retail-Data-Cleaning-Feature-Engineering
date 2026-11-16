"""
Unit tests for feature engineering operations.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Domain: Data Science
Task: Task 02 — Real-World Online Retail Data Cleaning & Feature Engineering
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    ALL_FEATURES,
    create_aggregation_features,
    create_transaction_features,
    engineer_all_features,
    get_feature_descriptions,
    validate_features,
)


@pytest.fixture
def sample_cleaned_data():
    """Create sample cleaned dataset for feature engineering tests."""
    data = {
        "InvoiceNo": ["536365", "536365", "C536379", "536381", "536382"],
        "StockCode": ["85123A", "71053", "D", "84406B", "22197"],
        "Description": ["WHITE HANGING HEART", "WHITE METAL LANTERN", "DISCOUNT", "CREAM CUPID", "POPCORN HOLDER"],
        "Quantity": [6, 2, -1, 15, 60],
        "InvoiceDate": pd.to_datetime([
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 09:41:00",
            "2010-12-04 14:11:00",  # Saturday -> IsWeekend = True
            "2011-05-15 11:22:00",  # Sunday -> IsWeekend = True
        ]),
        "UnitPrice": [2.55, 1.25, 27.50, 12.00, 55.00],
        "CustomerID": ["17850", "17850", "14527", "Unknown", "15311"],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "France", "Germany"],
    }
    return pd.DataFrame(data)


def test_total_amount_calculation(sample_cleaned_data):
    """Test TotalAmount formula: Quantity * UnitPrice."""
    featured = create_transaction_features(sample_cleaned_data)
    assert "TotalAmount" in featured.columns
    # Row 0: 6 * 2.55 = 15.30
    assert np.isclose(featured.loc[0, "TotalAmount"], 15.30)
    # Row 2 (cancellation): -1 * 27.50 = -27.50
    assert np.isclose(featured.loc[2, "TotalAmount"], -27.50)


def test_date_features_extraction(sample_cleaned_data):
    """Test extraction of year, month, day, hour, and day of week."""
    featured = create_transaction_features(sample_cleaned_data)

    assert featured.loc[0, "InvoiceYear"] == 2010
    assert featured.loc[0, "InvoiceMonth"] == 12
    assert featured.loc[0, "InvoiceDay"] == 1
    assert featured.loc[0, "InvoiceHour"] == 8
    assert featured.loc[0, "PurchaseDayOfWeek"] == 2  # Wednesday (0=Mon, 1=Tue, 2=Wed)


def test_cancellation_and_return_flags(sample_cleaned_data):
    """Test IsCancellation and IsReturn boolean identification."""
    featured = create_transaction_features(sample_cleaned_data)

    assert featured.loc[0, "IsCancellation"] == False
    assert featured.loc[0, "IsReturn"] == False

    assert featured.loc[2, "IsCancellation"] == True
    assert featured.loc[2, "IsReturn"] == True


def test_weekend_flag(sample_cleaned_data):
    """Test IsWeekend identifies Saturday and Sunday."""
    featured = create_transaction_features(sample_cleaned_data)

    assert featured.loc[0, "IsWeekend"] == False  # Wednesday
    assert featured.loc[3, "IsWeekend"] == True   # Saturday
    assert featured.loc[4, "IsWeekend"] == True   # Sunday


def test_quantity_and_price_categories(sample_cleaned_data):
    """Test binning into QuantityCategory and PriceCategory."""
    featured = create_transaction_features(sample_cleaned_data)

    # Qty: 6 -> Medium (3-12), 2 -> Low (0-2), 15 -> High (13-50), 60 -> Very High (>50)
    assert featured.loc[0, "QuantityCategory"] == "Medium"
    assert featured.loc[1, "QuantityCategory"] == "Low"
    assert featured.loc[3, "QuantityCategory"] == "High"
    assert featured.loc[4, "QuantityCategory"] == "Very High"

    # Price: 2.55 -> Standard (2-10), 1.25 -> Budget (0-2), 12.00 -> Premium (10-50), 55.00 -> Luxury (>50)
    assert featured.loc[0, "PriceCategory"] == "Standard"
    assert featured.loc[1, "PriceCategory"] == "Budget"
    assert featured.loc[3, "PriceCategory"] == "Premium"
    assert featured.loc[4, "PriceCategory"] == "Luxury"


def test_aggregation_features(sample_cleaned_data):
    """Test CustomerPurchaseFrequency, InvoiceItemCount, and AverageInvoiceValue."""
    df_trans = create_transaction_features(sample_cleaned_data)
    df_agg = create_aggregation_features(df_trans)

    # Invoice 536365 has 2 line items
    inv_536365 = df_agg[df_agg["InvoiceNo"] == "536365"]
    assert inv_536365.iloc[0]["InvoiceItemCount"] == 2

    # Average invoice value for 536365: (6*2.55 + 2*1.25)/2 = (15.30 + 2.50)/2 = 8.90
    assert np.isclose(inv_536365.iloc[0]["AverageInvoiceValue"], 8.90)

    # Customer 17850 has 1 unique invoice in this sample
    assert inv_536365.iloc[0]["CustomerPurchaseFrequency"] == 1


def test_engineer_all_features_pipeline(sample_cleaned_data):
    """Verify that engineer_all_features creates all 14 required features."""
    featured = engineer_all_features(sample_cleaned_data)

    for feat in ALL_FEATURES:
        assert feat in featured.columns, f"Feature '{feat}' missing from engineered dataset"

    validation = validate_features(featured)
    assert validation["passed"] == True
    assert len(validation["checks"]) == 14


def test_feature_descriptions_complete():
    """Verify that all features have clear documentation descriptions."""
    descriptions = get_feature_descriptions()
    for feat in ALL_FEATURES:
        assert feat in descriptions, f"Feature '{feat}' lacks a description"
        assert len(descriptions[feat]) > 10

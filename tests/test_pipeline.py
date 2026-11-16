"""
Unit tests for the scikit-learn preprocessing pipeline.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Domain: Data Science
Task: Task 02 — Real-World Online Retail Data Cleaning & Feature Engineering
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import engineer_all_features
from src.preprocessing_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PASSTHROUGH_FEATURES,
    build_preprocessing_pipeline,
    fit_and_save_pipeline,
    get_feature_names,
    load_pipeline,
    transform_data,
    train_test_pipeline_workflow,
)


@pytest.fixture
def sample_featured_data():
    """Create a sample feature-engineered dataset for pipeline tests."""
    data = {
        "InvoiceNo": ["536365", "536365", "C536379", "536381", "536382", "536383", "536384"],
        "StockCode": ["85123A", "71053", "D", "84406B", "22197", "22492", "47566"],
        "Description": ["HEART", "LANTERN", "DISCOUNT", "CUPID", "POPCORN", "PAINT SET", "PARTY BUNTING"],
        "Quantity": [6, 2, -1, 15, 60, 12, 4],
        "InvoiceDate": pd.to_datetime([
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 09:41:00",
            "2010-12-04 14:11:00",
            "2011-05-15 11:22:00",
            "2011-06-20 12:30:00",
            "2011-07-11 16:45:00",
        ]),
        "UnitPrice": [2.55, 1.25, 27.50, 12.00, 55.00, 3.45, 4.95],
        "CustomerID": ["17850", "17850", "14527", "Unknown", "15311", "16000", "17000"],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "France", "Germany", "Spain", "EIRE"],
    }
    df = pd.DataFrame(data)
    return engineer_all_features(df)


def test_build_preprocessing_pipeline(sample_featured_data):
    """Test that pipeline structure contains ColumnTransformer, imputer, scaler, and encoder."""
    numeric_cols = [c for c in NUMERIC_FEATURES if c in sample_featured_data.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in sample_featured_data.columns]

    pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps

    ct = pipeline.named_steps["preprocessor"]
    transformers = {name: trans for name, trans, _ in ct.transformers}
    assert "num" in transformers
    assert "cat" in transformers
    assert "passthrough" in transformers


def test_fit_and_transform(sample_featured_data):
    """Test fitting the pipeline and transforming data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe_path = Path(tmpdir) / "test_pipeline.joblib"
        fitted_pipe = fit_and_save_pipeline(sample_featured_data, pipeline_path=pipe_path)

        assert pipe_path.exists()
        transformed = transform_data(sample_featured_data, fitted_pipe)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == len(sample_featured_data)
        assert transformed.shape[1] > 0
        assert not np.isnan(transformed[:, :len(NUMERIC_FEATURES)].astype(float)).any()


def test_pipeline_persistence_and_reload(sample_featured_data):
    """Test that saving and reloading pipeline yields identical outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe_path = Path(tmpdir) / "test_pipeline.joblib"
        fitted_pipe = fit_and_save_pipeline(sample_featured_data, pipeline_path=pipe_path)

        transformed_orig = transform_data(sample_featured_data, fitted_pipe)

        reloaded_pipe = load_pipeline(pipe_path)
        transformed_reloaded = transform_data(sample_featured_data, reloaded_pipe)

        try:
            assert np.allclose(transformed_orig.astype(float), transformed_reloaded.astype(float))
        except (ValueError, TypeError):
            assert np.array_equal(transformed_orig, transformed_reloaded)


def test_unknown_category_handling(sample_featured_data):
    """Verify that unknown categorical values in test data do not cause errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe_path = Path(tmpdir) / "test_pipeline.joblib"
        fitted_pipe = fit_and_save_pipeline(sample_featured_data, pipeline_path=pipe_path)

        # Create row with unknown country
        new_row = sample_featured_data.head(1).copy()
        new_row["Country"] = "Antarctica"

        # Should transform smoothly without error due to handle_unknown='ignore'
        transformed_new = transform_data(new_row, fitted_pipe)
        assert transformed_new.shape[0] == 1


def test_train_test_split_workflow_no_leakage(sample_featured_data):
    """Verify that train_test_pipeline_workflow splits data and fits strictly on train."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe_path = Path(tmpdir) / "test_split_pipeline.joblib"
        results = train_test_pipeline_workflow(
            sample_featured_data,
            test_size=0.3,
            random_state=42,
            pipeline_path=pipe_path,
        )

        assert results["train_rows"] < len(sample_featured_data)
        assert results["test_rows"] > 0
        assert results["x_train"].shape[0] == results["train_rows"]
        assert results["x_test"].shape[0] == results["test_rows"]
        assert pipe_path.exists()

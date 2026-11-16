"""
Preprocessing Pipeline module for Online Retail Data Cleaning & Feature Engineering.

Builds a scikit-learn preprocessing pipeline that handles numeric scaling
and categorical encoding for the cleaned and feature-engineered Online Retail
dataset.  The fitted pipeline can be persisted with joblib and reloaded for
inference.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    IDENTIFIER_COLUMNS,
    MODELS_DIR,
    PIPELINE_FILE,
    RANDOM_SEED,
)

# ── Column groups after feature engineering ──────────────────────────────────

NUMERIC_FEATURES: List[str] = [
    "Quantity",
    "UnitPrice",
    "TotalAmount",
    "InvoiceItemCount",
    "AverageInvoiceValue",
    "CustomerPurchaseFrequency",
]

CATEGORICAL_FEATURES: List[str] = [
    "Country",
    "QuantityCategory",
    "PriceCategory",
    "PurchaseDayOfWeek",
]

BOOLEAN_FEATURES: List[str] = [
    "IsCancellation",
    "IsReturn",
    "IsWeekend",
]

DATE_PART_FEATURES: List[str] = [
    "InvoiceYear",
    "InvoiceMonth",
    "InvoiceDay",
    "InvoiceHour",
]

# Passthrough columns (booleans + date parts treated as numeric pass-through)
PASSTHROUGH_FEATURES: List[str] = BOOLEAN_FEATURES + DATE_PART_FEATURES


# =============================================================================
# Pipeline Construction
# =============================================================================


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """Build a scikit-learn preprocessing pipeline.

    Parameters
    ----------
    numeric_features : list[str]
        Column names to apply median imputation and standard scaling.
    categorical_features : list[str]
        Column names to apply most-frequent imputation and one-hot encoding.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline wrapping a ``ColumnTransformer``.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("passthrough", "passthrough", PASSTHROUGH_FEATURES),
        ],
        remainder="drop",  # drop identifier / unused columns
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor)],
    )

    return pipeline


# =============================================================================
# Fit / Save / Load
# =============================================================================


def fit_and_save_pipeline(
    df: pd.DataFrame,
    pipeline_path: Optional[Path] = None,
) -> Pipeline:
    """Fit the preprocessing pipeline on *df* and persist it with joblib.

    The function automatically selects the numeric and categorical features
    defined at module level and excludes identifier columns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset to fit on.
    pipeline_path : Path, optional
        Destination for the joblib file.  Defaults to ``config.PIPELINE_FILE``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline.
    """

    if pipeline_path is None:
        pipeline_path = PIPELINE_FILE

    pipeline_path = Path(pipeline_path)

    # Filter to columns actually present in the dataframe
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)

    # Only keep columns used by the pipeline (exclude identifiers, etc.)
    cols_to_use = numeric_cols + categorical_cols + [
        c for c in PASSTHROUGH_FEATURES if c in df.columns
    ]
    pipeline.fit(df[cols_to_use])

    # Ensure output directory exists
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Pipeline fitted and saved to {pipeline_path}")

    return pipeline


def load_pipeline(pipeline_path: Optional[Path] = None) -> Pipeline:
    """Load a previously saved preprocessing pipeline.

    Parameters
    ----------
    pipeline_path : Path, optional
        Path to the joblib file.  Defaults to ``config.PIPELINE_FILE``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded (fitted) pipeline.

    Raises
    ------
    FileNotFoundError
        If the pipeline file does not exist.
    """

    if pipeline_path is None:
        pipeline_path = PIPELINE_FILE

    pipeline_path = Path(pipeline_path)

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline file not found at {pipeline_path}. "
            "Run fit_and_save_pipeline() first."
        )

    pipeline = joblib.load(pipeline_path)
    print(f"✅ Pipeline loaded from {pipeline_path}")
    return pipeline


# =============================================================================
# Transform & Feature Names
# =============================================================================


def transform_data(df: pd.DataFrame, pipeline: Pipeline) -> np.ndarray:
    """Transform *df* using a fitted pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns the pipeline was fitted on.
    pipeline : sklearn.pipeline.Pipeline
        A fitted preprocessing pipeline.

    Returns
    -------
    np.ndarray
        The transformed feature matrix.
    """

    # Identify columns the preprocessor expects
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    expected_cols: List[str] = []
    for _, _, cols in preprocessor.transformers:
        if isinstance(cols, list):
            expected_cols.extend(cols)

    cols_present = [c for c in expected_cols if c in df.columns]
    transformed = pipeline.transform(df[cols_present])
    return transformed


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Return the output feature names from a fitted pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted preprocessing pipeline.

    Returns
    -------
    list[str]
        Ordered list of output feature names.
    """

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


# =============================================================================
# Train / Test Workflow (Prevents Data Leakage)
# =============================================================================


def train_test_pipeline_workflow(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    pipeline_path: Optional[Path] = None,
) -> dict:
    """Split data into train and test sets, fit pipeline strictly on train data,
    and transform both sets to ensure zero data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned feature-engineered dataset.
    test_size : float, default 0.2
        Fraction of data reserved for testing.
    random_state : int, default 42
        Reproducibility seed.
    pipeline_path : Path, optional
        Destination for the saved joblib pipeline.

    Returns
    -------
    dict
        Dictionary containing train/test shapes, feature names, transformed arrays,
        and pipeline object.
    """
    if pipeline_path is None:
        pipeline_path = PIPELINE_FILE

    print("\n--- Train / Test Pipeline Workflow (Zero Data Leakage) ---")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"  Training set size : {len(train_df):,} rows ({(1 - test_size) * 100:.0f}%)")
    print(f"  Testing set size  : {len(test_df):,} rows ({test_size * 100:.0f}%)")

    # Fit pipeline strictly on train_df
    pipeline = fit_and_save_pipeline(train_df, pipeline_path=pipeline_path)

    # Transform both splits
    x_train_transformed = transform_data(train_df, pipeline)
    x_test_transformed = transform_data(test_df, pipeline)
    feature_names = get_feature_names(pipeline)

    print(f"  Transformed X_train shape: {x_train_transformed.shape}")
    print(f"  Transformed X_test shape : {x_test_transformed.shape}")
    print(f"  Total processed features : {len(feature_names)}")

    return {
        "pipeline": pipeline,
        "x_train": x_train_transformed,
        "x_test": x_test_transformed,
        "feature_names": feature_names,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }


# =============================================================================
# Demonstration
# =============================================================================


def demonstrate_pipeline(df: pd.DataFrame) -> None:
    """Demonstrate the full pipeline lifecycle on *df*.

    Splits data, fits pipeline strictly on training data (no leakage),
    saves to models/, reloads, and transforms test data.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset.
    """

    print("=" * 70)
    print("PREPROCESSING PIPELINE DEMONSTRATION")
    print("=" * 70)

    # 1. Train / Test Workflow
    workflow_results = train_test_pipeline_workflow(df)
    pipeline = workflow_results["pipeline"]
    feature_names = workflow_results["feature_names"]

    # 2. Output features sample
    print(f"\n📋 Output features sample (first 15 of {len(feature_names)}):")
    for i, name in enumerate(feature_names[:15], 1):
        print(f"   {i:2d}. {name}")

    # 3. Transform sample check
    sample = df.head(5)
    transformed = transform_data(sample, pipeline)
    print(f"\n🔄 Sample transformation check (5 rows): shape = {transformed.shape}")

    # 4. Reload round-trip check
    print("\n🔁 Reload round-trip check from disk …")
    reloaded = load_pipeline()
    rt_transformed = transform_data(sample, reloaded)
    try:
        matches = np.allclose(transformed.astype(float), rt_transformed.astype(float))
    except (ValueError, TypeError):
        matches = np.array_equal(transformed, rt_transformed)

    if matches:
        print("   ✅ Round-trip successful — saved pipeline produces identical outputs.")
    else:
        print("   ❌ Round-trip mismatch!")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    from src.config import FEATURE_ENGINEERED_FILE, ensure_directories

    ensure_directories()

    if FEATURE_ENGINEERED_FILE.exists():
        print(f"Loading feature-engineered data from {FEATURE_ENGINEERED_FILE} …")
        df = pd.read_csv(FEATURE_ENGINEERED_FILE)
        demonstrate_pipeline(df)
    else:
        print(
            f"⚠️  Feature-engineered file not found at {FEATURE_ENGINEERED_FILE}.\n"
            "   Run the feature engineering step first."
        )

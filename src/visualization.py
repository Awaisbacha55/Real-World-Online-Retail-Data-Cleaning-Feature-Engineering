"""
Visualization module for Online Retail Data Cleaning & Feature Engineering.

Provides 14+ publication-ready visualizations covering data quality,
distributions, outliers, trends, and the results of feature engineering.
All plots use a consistent *whitegrid* Seaborn style and are saved at
150 DPI to ``outputs/figures/`` by default.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CURRENCY_SYMBOL,
    FIGURE_DPI,
    FIGURE_SIZE_STANDARD,
    FIGURE_SIZE_TALL,
    FIGURE_SIZE_WIDE,
    FIGURES_DIR,
    STYLE,
)

# ── Apply global style ──────────────────────────────────────────────────────
sns.set_style(STYLE)


# ── Helper ───────────────────────────────────────────────────────────────────


def _save_figure(fig: plt.Figure, save_path: Optional[Path], default_name: str) -> None:
    """Save *fig* to *save_path* (or ``FIGURES_DIR / default_name``)."""
    if save_path is None:
        save_path = FIGURES_DIR / default_name
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved: {save_path}")


# =============================================================================
# 1. Missing Values
# =============================================================================


def plot_missing_values(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Bar chart of missing-value counts and percentages per column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or intermediate dataset.
    save_path : Path, optional
        Override default save location.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("  ℹ️  No missing values to plot.")
        return

    pct = (missing / len(df) * 100).round(2)

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_STANDARD)

    # Counts
    axes[0].barh(missing.index, missing.values, color=sns.color_palette("viridis", len(missing)))
    axes[0].set_xlabel("Missing Count")
    axes[0].set_title("Missing Values — Count")
    for i, (v, p) in enumerate(zip(missing.values, pct.values)):
        axes[0].text(v + len(df) * 0.005, i, f"{v:,}", va="center", fontsize=9)

    # Percentages
    axes[1].barh(pct.index, pct.values, color=sns.color_palette("magma", len(pct)))
    axes[1].set_xlabel("Missing (%)")
    axes[1].set_title("Missing Values — Percentage")
    for i, p in enumerate(pct.values):
        axes[1].text(p + 0.3, i, f"{p:.2f}%", va="center", fontsize=9)

    fig.suptitle("Missing Value Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, save_path, "missing_values.png")


# =============================================================================
# 2. Duplicate Analysis
# =============================================================================


def plot_duplicate_analysis(
    before_count: int,
    after_count: int,
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing row counts before and after duplicate removal.

    Parameters
    ----------
    before_count : int
        Number of rows before dedup.
    after_count : int
        Number of rows after dedup.
    save_path : Path, optional
        Override default save location.
    """
    removed = before_count - after_count

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    labels = ["Before Cleaning", "After Cleaning", "Duplicates Removed"]
    values = [before_count, after_count, removed]
    colours = ["#3498db", "#2ecc71", "#e74c3c"]

    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + before_count * 0.01,
            f"{val:,}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Number of Rows")
    ax.set_title("Duplicate Analysis — Before vs After Cleaning", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path, "duplicate_analysis.png")


# =============================================================================
# 3 & 4. Distribution Histograms
# =============================================================================


def plot_quantity_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Histogram of the *Quantity* column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a ``Quantity`` column.
    save_path : Path, optional
        Override default save location.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    data = df["Quantity"].dropna()
    ax.hist(data, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Quantity", fontsize=13, fontweight="bold")
    ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.1f}")
    ax.axvline(data.median(), color="green", linestyle="--", label=f"Median: {data.median():.1f}")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path, "quantity_distribution.png")


def plot_unitprice_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Histogram of the *UnitPrice* column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a ``UnitPrice`` column.
    save_path : Path, optional
        Override default save location.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    data = df["UnitPrice"].dropna()
    ax.hist(data, bins=50, color="#e67e22", edgecolor="white", alpha=0.85)
    ax.set_xlabel(f"Unit Price ({CURRENCY_SYMBOL})")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Unit Price ({CURRENCY_SYMBOL})", fontsize=13, fontweight="bold")
    ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {CURRENCY_SYMBOL}{data.mean():.2f}")
    ax.axvline(data.median(), color="green", linestyle="--", label=f"Median: {CURRENCY_SYMBOL}{data.median():.2f}")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path, "unitprice_distribution.png")


# =============================================================================
# 5 & 6. Box Plots
# =============================================================================


def plot_quantity_boxplot(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Boxplot for *Quantity* to visualise outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a ``Quantity`` column.
    save_path : Path, optional
        Override default save location.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    sns.boxplot(x=df["Quantity"], ax=ax, color="#3498db", flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.set_xlabel("Quantity")
    ax.set_title("Quantity — Boxplot (Outlier Detection)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path, "quantity_boxplot.png")


def plot_unitprice_boxplot(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Boxplot for *UnitPrice* to visualise outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a ``UnitPrice`` column.
    save_path : Path, optional
        Override default save location.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    sns.boxplot(x=df["UnitPrice"], ax=ax, color="#e67e22", flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.set_xlabel(f"Unit Price ({CURRENCY_SYMBOL})")
    ax.set_title(f"Unit Price — Boxplot (Outlier Detection)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path, "unitprice_boxplot.png")


# =============================================================================
# 7. TotalAmount Distribution
# =============================================================================


def plot_total_amount_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Histogram of *TotalAmount* (Quantity × UnitPrice).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a ``TotalAmount`` column.
    save_path : Path, optional
        Override default save location.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    data = df["TotalAmount"].dropna()
    ax.hist(data, bins=50, color="#2ecc71", edgecolor="white", alpha=0.85)
    ax.set_xlabel(f"Total Amount ({CURRENCY_SYMBOL})")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Total Amount ({CURRENCY_SYMBOL})", fontsize=13, fontweight="bold")
    ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {CURRENCY_SYMBOL}{data.mean():.2f}")
    ax.axvline(data.median(), color="green", linestyle="--", label=f"Median: {CURRENCY_SYMBOL}{data.median():.2f}")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path, "total_amount_distribution.png")


# =============================================================================
# 8. Country Transactions
# =============================================================================


def plot_country_transactions(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Horizontal bar chart of the top 15 countries by transaction count.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a ``Country`` column.
    save_path : Path, optional
        Override default save location.
    """
    top_countries = df["Country"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    top_countries.sort_values().plot.barh(
        ax=ax,
        color=sns.color_palette("viridis", len(top_countries)),
    )
    ax.set_xlabel("Number of Transactions")
    ax.set_ylabel("Country")
    ax.set_title("Top 15 Countries by Transaction Count", fontsize=13, fontweight="bold")

    for i, v in enumerate(top_countries.sort_values().values):
        ax.text(v + top_countries.max() * 0.01, i, f"{v:,}", va="center", fontsize=9)

    fig.tight_layout()
    _save_figure(fig, save_path, "country_transactions.png")


# =============================================================================
# 9. Monthly Trends
# =============================================================================


def plot_monthly_trends(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Line plot of monthly transaction counts.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing an ``InvoiceDate`` column (datetime or string).
    save_path : Path, optional
        Override default save location.
    """
    temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp["InvoiceDate"]):
        temp["InvoiceDate"] = pd.to_datetime(temp["InvoiceDate"], errors="coerce")

    try:
        monthly = temp.set_index("InvoiceDate").resample("ME").size()
    except ValueError:
        monthly = temp.set_index("InvoiceDate").resample("M").size()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.plot(monthly.index, monthly.values, marker="o", color="#2c3e50", linewidth=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#3498db")
    ax.set_xlabel("Month")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Monthly Transaction Trends", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45)
    fig.tight_layout()
    _save_figure(fig, save_path, "monthly_trends.png")


# =============================================================================
# 10. Cancellation Analysis
# =============================================================================


def plot_cancellation_analysis(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Pie and bar chart of cancellation vs. normal transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``IsCancellation`` boolean column **or** ``InvoiceNo``
        where cancellations start with 'C'.
    save_path : Path, optional
        Override default save location.
    """
    if "IsCancellation" in df.columns:
        cancel_counts = df["IsCancellation"].value_counts()
        labels = ["Normal", "Cancellation"]
    else:
        is_cancel = df["InvoiceNo"].astype(str).str.startswith("C")
        cancel_counts = is_cancel.value_counts()
        labels = ["Normal", "Cancellation"]

    values = [cancel_counts.get(False, 0), cancel_counts.get(True, 0)]
    colours = ["#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_STANDARD)

    # Pie
    axes[0].pie(values, labels=labels, colors=colours, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("Cancellation Ratio")

    # Bar
    axes[1].bar(labels, values, color=colours, edgecolor="white")
    for i, v in enumerate(values):
        axes[1].text(i, v + max(values) * 0.01, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Cancellation Counts")

    fig.suptitle("Cancellation Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, save_path, "cancellation_analysis.png")


# =============================================================================
# 11. Returns Analysis
# =============================================================================


def plot_returns_analysis(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Bar chart of return vs. normal transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``IsReturn`` boolean column **or** ``Quantity`` column
        (negative quantities treated as returns).
    save_path : Path, optional
        Override default save location.
    """
    if "IsReturn" in df.columns:
        return_counts = df["IsReturn"].value_counts()
    else:
        return_counts = (df["Quantity"] < 0).value_counts()

    labels = ["Normal", "Return"]
    values = [return_counts.get(False, 0), return_counts.get(True, 0)]
    colours = ["#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    bars = ax.bar(labels, values, color=colours, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:,}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Count")
    ax.set_title("Returns Analysis — Normal vs Returns", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path, "returns_analysis.png")


# =============================================================================
# 12. Correlation Heatmap
# =============================================================================


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Heatmap of correlations among numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with numeric columns.
    save_path : Path, optional
        Override default save location.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_TALL)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlation Heatmap — Numeric Features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path, "correlation_heatmap.png")


# =============================================================================
# 13. Before / After Comparison
# =============================================================================


def plot_before_after_comparison(
    before_stats: Dict[str, float],
    after_stats: Dict[str, float],
    save_path: Optional[Path] = None,
) -> None:
    """Grouped bar chart comparing summary statistics before and after cleaning.

    Parameters
    ----------
    before_stats : dict[str, float]
        Metric name → value before cleaning (e.g. ``{"Rows": 541909, …}``).
    after_stats : dict[str, float]
        Metric name → value after cleaning.
    save_path : Path, optional
        Override default save location.
    """
    metrics = list(before_stats.keys())
    before_vals = [before_stats[m] for m in metrics]
    after_vals = [after_stats[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    bars1 = ax.bar(x - width / 2, before_vals, width, label="Before Cleaning", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, after_vals, width, label="After Cleaning", color="#2ecc71", alpha=0.85)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Before vs After Cleaning — Key Statistics", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.legend()

    # Annotate
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():,.0f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():,.0f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save_figure(fig, save_path, "before_after_comparison.png")


# =============================================================================
# 14. Feature Engineering Results
# =============================================================================


def plot_feature_engineering_results(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Multi-panel visualisation of engineered feature distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset.
    save_path : Path, optional
        Override default save location.
    """
    # Pick features that exist in the dataframe
    feature_specs = [
        ("TotalAmount", f"Total Amount ({CURRENCY_SYMBOL})", "#2ecc71", "hist"),
        ("CustomerPurchaseFrequency", "Customer Purchase Frequency", "#3498db", "hist"),
        ("InvoiceItemCount", "Invoice Item Count", "#e67e22", "hist"),
        ("AverageInvoiceValue", f"Avg Invoice Value ({CURRENCY_SYMBOL})", "#9b59b6", "hist"),
        ("QuantityCategory", "Quantity Category", "#1abc9c", "count"),
        ("PriceCategory", "Price Category", "#e74c3c", "count"),
    ]

    available = [(name, label, colour, kind) for name, label, colour, kind in feature_specs if name in df.columns]

    if not available:
        print("  ℹ️  No engineered features found to plot.")
        return

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (name, label, colour, kind) in enumerate(available):
        ax = axes[idx]
        if kind == "hist":
            data = df[name].dropna()
            ax.hist(data, bins=30, color=colour, edgecolor="white", alpha=0.85)
            ax.axvline(data.mean(), color="red", linestyle="--", linewidth=0.8)
        else:
            counts = df[name].value_counts()
            counts.plot.bar(ax=ax, color=colour, edgecolor="white")
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency")

    # Hide unused subplots
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Engineering Results", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, save_path, "feature_engineering_results.png")


# =============================================================================
# 15. Master Function
# =============================================================================


def create_all_visualizations(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    before_count: int,
    after_count: int,
) -> None:
    """Generate and save every visualisation in one call.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original raw dataset.
    cleaned_df : pd.DataFrame
        Cleaned dataset (after dedup, type fixes, etc.).
    featured_df : pd.DataFrame
        Dataset after feature engineering.
    before_count : int
        Row count before duplicate removal.
    after_count : int
        Row count after duplicate removal.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 70)

    print("\n📊 1/14  Missing Values …")
    plot_missing_values(raw_df)

    print("📊 2/14  Duplicate Analysis …")
    plot_duplicate_analysis(before_count, after_count)

    print("📊 3/14  Quantity Distribution …")
    plot_quantity_distribution(cleaned_df)

    print("📊 4/14  Unit Price Distribution …")
    plot_unitprice_distribution(cleaned_df)

    print("📊 5/14  Quantity Boxplot …")
    plot_quantity_boxplot(cleaned_df)

    print("📊 6/14  Unit Price Boxplot …")
    plot_unitprice_boxplot(cleaned_df)

    print("📊 7/14  Total Amount Distribution …")
    plot_total_amount_distribution(featured_df)

    print("📊 8/14  Country Transactions …")
    plot_country_transactions(cleaned_df)

    print("📊 9/14  Monthly Trends …")
    plot_monthly_trends(cleaned_df)

    print("📊 10/14 Cancellation Analysis …")
    plot_cancellation_analysis(featured_df)

    print("📊 11/14 Returns Analysis …")
    plot_returns_analysis(featured_df)

    print("📊 12/14 Correlation Heatmap …")
    plot_correlation_heatmap(featured_df)

    print("📊 13/14 Before vs After Comparison …")
    # Build quick stats dicts from raw & cleaned
    before_stats = {
        "Rows": len(raw_df),
        "Columns": len(raw_df.columns),
        "Missing Values": int(raw_df.isnull().sum().sum()),
        "Duplicates": before_count - after_count,
    }
    after_stats = {
        "Rows": len(cleaned_df),
        "Columns": len(cleaned_df.columns),
        "Missing Values": int(cleaned_df.isnull().sum().sum()),
        "Duplicates": 0,
    }
    plot_before_after_comparison(before_stats, after_stats)

    print("📊 14/14 Feature Engineering Results …")
    plot_feature_engineering_results(featured_df)

    print("\n" + "=" * 70)
    print(f"✅ ALL VISUALIZATIONS SAVED to {FIGURES_DIR}")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    from src.config import (
        CLEANED_DATA_FILE,
        FEATURE_ENGINEERED_FILE,
        RAW_DATA_FILE,
        ensure_directories,
    )

    ensure_directories()

    # Attempt to load available datasets
    print("Loading datasets for visualization …\n")

    dfs: dict[str, pd.DataFrame | None] = {"raw": None, "cleaned": None, "featured": None}

    if RAW_DATA_FILE.exists():
        dfs["raw"] = pd.read_excel(RAW_DATA_FILE)
        print(f"  ✅ Raw data loaded: {dfs['raw'].shape}")
    else:
        print(f"  ⚠️  Raw data not found at {RAW_DATA_FILE}")

    if CLEANED_DATA_FILE.exists():
        dfs["cleaned"] = pd.read_csv(CLEANED_DATA_FILE)
        print(f"  ✅ Cleaned data loaded: {dfs['cleaned'].shape}")
    else:
        print(f"  ⚠️  Cleaned data not found at {CLEANED_DATA_FILE}")

    if FEATURE_ENGINEERED_FILE.exists():
        dfs["featured"] = pd.read_csv(FEATURE_ENGINEERED_FILE)
        print(f"  ✅ Featured data loaded: {dfs['featured'].shape}")
    else:
        print(f"  ⚠️  Featured data not found at {FEATURE_ENGINEERED_FILE}")

    # Generate whatever we can
    if all(v is not None for v in dfs.values()):
        raw_count = len(dfs["raw"])  # type: ignore[arg-type]
        cleaned_count = len(dfs["cleaned"])  # type: ignore[arg-type]
        create_all_visualizations(
            dfs["raw"], dfs["cleaned"], dfs["featured"],  # type: ignore[arg-type]
            before_count=raw_count,
            after_count=cleaned_count,
        )
    else:
        # Generate individual plots for whatever we have
        print("\n⚠️  Not all datasets available — generating individual plots.\n")
        if dfs["raw"] is not None:
            plot_missing_values(dfs["raw"])
        if dfs["cleaned"] is not None:
            plot_quantity_distribution(dfs["cleaned"])
            plot_unitprice_distribution(dfs["cleaned"])
            plot_quantity_boxplot(dfs["cleaned"])
            plot_unitprice_boxplot(dfs["cleaned"])
            plot_country_transactions(dfs["cleaned"])
            plot_monthly_trends(dfs["cleaned"])
        if dfs["featured"] is not None:
            plot_total_amount_distribution(dfs["featured"])
            plot_cancellation_analysis(dfs["featured"])
            plot_returns_analysis(dfs["featured"])
            plot_correlation_heatmap(dfs["featured"])
            plot_feature_engineering_results(dfs["featured"])

    print("\nDone.")

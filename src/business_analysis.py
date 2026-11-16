"""
Business Analysis module for Online Retail Data Cleaning & Feature Engineering.

Conducts in-depth commercial and exploratory business analysis on the cleaned
and feature-engineered UCI Online Retail dataset according to Devixo Solutions
Task 02 guidelines:
- Top 10 products by quantity sold and revenue generated
- Top 10 countries by transaction volume and monetary turnover
- Monthly sales trends, revenue trajectory, and seasonal patterns
- Average Order Value (AOV) and basket size metrics
- Commercial cancellation patterns, return rates, and impacted products
- Customer purchasing frequency distribution (repeat buyer vs one-time)
- Revenue distribution and Pareto principle (80/20 rule) assessment
- Comprehensive business insights reporting and executive summary

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
    BUSINESS_INSIGHTS_FILE,
    CANCELLATION_PREFIX,
    CURRENCY_LABEL,
    CURRENCY_SYMBOL,
    FEATURE_ENGINEERED_FILE,
    POSITIVE_SALES_FILE,
    REPORTS_DIR,
    ensure_directories,
)


# =============================================================================
# 1. Top Products Analysis
# =============================================================================

def get_top_products(
    df: pd.DataFrame,
    top_n: int = 10,
    by: str = "Quantity",
) -> pd.DataFrame:
    """Identify top selling products by quantity sold or revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset (preferably positive sales or feature-engineered transactions).
    top_n : int, default 10
        Number of top products to return.
    by : {'Quantity', 'TotalAmount'}, default 'Quantity'
        Metric to rank products by.

    Returns
    -------
    pd.DataFrame
        Summary table with Description, StockCode, Quantity, TotalAmount,
        and TransactionCount.
    """
    # Filter positive transactions for product demand ranking
    valid_sales = df[df["Quantity"] > 0].copy()

    # Ensure TotalAmount exists
    if "TotalAmount" not in valid_sales.columns:
        valid_sales["TotalAmount"] = valid_sales["Quantity"] * valid_sales["UnitPrice"]

    grouped = valid_sales.groupby(["StockCode", "Description"]).agg(
        TotalQuantity=("Quantity", "sum"),
        TotalRevenue=("TotalAmount", "sum"),
        TransactionCount=("InvoiceNo", "count"),
        AverageUnitPrice=("UnitPrice", "mean"),
    ).reset_index()

    sort_col = "TotalQuantity" if by == "Quantity" else "TotalRevenue"
    top_df = grouped.sort_values(by=sort_col, ascending=False).head(top_n).reset_index(drop=True)

    top_df["TotalRevenue"] = top_df["TotalRevenue"].round(2)
    top_df["AverageUnitPrice"] = top_df["AverageUnitPrice"].round(2)

    return top_df


# =============================================================================
# 2. Top Countries Analysis
# =============================================================================

def get_top_countries(
    df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Analyze geographic concentration of transactions and revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing Country, Quantity, and UnitPrice/TotalAmount.
    top_n : int, default 10
        Number of top countries to return.

    Returns
    -------
    pd.DataFrame
        Summary table with Country, TransactionCount, TotalQuantity,
        TotalRevenue, UniqueCustomers, and RevenueSharePct.
    """
    data = df.copy()
    if "TotalAmount" not in data.columns:
        data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

    # Filter positive transactions for revenue calculations
    positive_data = data[data["TotalAmount"] > 0]
    total_global_revenue = positive_data["TotalAmount"].sum()

    grouped = positive_data.groupby("Country").agg(
        TransactionCount=("InvoiceNo", "count"),
        TotalQuantity=("Quantity", "sum"),
        TotalRevenue=("TotalAmount", "sum"),
        UniqueCustomers=("CustomerID", lambda s: s[s != "Unknown"].nunique()),
    ).reset_index()

    grouped["RevenueSharePct"] = (
        grouped["TotalRevenue"] / total_global_revenue * 100
    ).round(2)
    grouped["TotalRevenue"] = grouped["TotalRevenue"].round(2)

    top_countries = grouped.sort_values(by="TransactionCount", ascending=False).head(top_n).reset_index(drop=True)
    return top_countries


# =============================================================================
# 3. Monthly Sales Trends
# =============================================================================

def get_monthly_sales_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly revenue, order count, and average order value.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing InvoiceDate, TotalAmount, InvoiceNo, and Quantity.

    Returns
    -------
    pd.DataFrame
        Monthly metrics with YearMonth, Revenue, OrderCount, UnitsSold,
        and MonthlyAOV.
    """
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["InvoiceDate"]):
        data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

    if "TotalAmount" not in data.columns:
        data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

    # Use completed positive transactions
    pos_data = data[data["TotalAmount"] > 0].copy()
    pos_data["YearMonth"] = pos_data["InvoiceDate"].dt.to_period("M").astype(str)

    monthly = pos_data.groupby("YearMonth").agg(
        Revenue=("TotalAmount", "sum"),
        OrderCount=("InvoiceNo", "nunique"),
        UnitsSold=("Quantity", "sum"),
        TransactionLines=("InvoiceNo", "count"),
    ).reset_index()

    monthly["MonthlyAOV"] = (monthly["Revenue"] / monthly["OrderCount"]).round(2)
    monthly["Revenue"] = monthly["Revenue"].round(2)

    return monthly


# =============================================================================
# 4. Average Order Value (AOV)
# =============================================================================

def calculate_average_order_value(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall Average Order Value and order-level distribution statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing InvoiceNo and TotalAmount.

    Returns
    -------
    dict
        Mean AOV, Median AOV, Standard Deviation, Min, Max, 25th and 75th percentiles.
    """
    data = df.copy()
    if "TotalAmount" not in data.columns:
        data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

    # Filter positive sales
    pos_sales = data[data["TotalAmount"] > 0]
    invoice_totals = pos_sales.groupby("InvoiceNo")["TotalAmount"].sum()

    return {
        "mean_aov": round(float(invoice_totals.mean()), 2),
        "median_aov": round(float(invoice_totals.median()), 2),
        "std_aov": round(float(invoice_totals.std()), 2),
        "min_order": round(float(invoice_totals.min()), 2),
        "max_order": round(float(invoice_totals.max()), 2),
        "q25_aov": round(float(invoice_totals.quantile(0.25)), 2),
        "q75_aov": round(float(invoice_totals.quantile(0.75)), 2),
        "total_orders": int(len(invoice_totals)),
    }


# =============================================================================
# 5. Cancellation Patterns Analysis
# =============================================================================

def analyze_cancellations(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze cancellation patterns, rates, and impacted products.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with InvoiceNo, Quantity, UnitPrice, and Country.

    Returns
    -------
    dict
        Cancellation count, percentage of all invoices, gross cancelled revenue,
        and top cancelled items.
    """
    data = df.copy()
    if "IsCancellation" in data.columns:
        cancel_mask = data["IsCancellation"] == True
    else:
        cancel_mask = data["InvoiceNo"].astype(str).str.startswith(CANCELLATION_PREFIX)

    if "TotalAmount" not in data.columns:
        data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

    total_invoices = data["InvoiceNo"].nunique()
    cancel_invoices = data[cancel_mask]["InvoiceNo"].nunique()
    cancel_invoice_rate = round(cancel_invoices / total_invoices * 100, 2) if total_invoices else 0.0

    total_rows = len(data)
    cancel_rows = int(cancel_mask.sum())
    cancel_row_pct = round(cancel_rows / total_rows * 100, 2) if total_rows else 0.0

    cancelled_revenue = abs(float(data[cancel_mask]["TotalAmount"].sum()))

    # Top cancelled products
    top_cancelled_products = (
        data[cancel_mask]
        .groupby(["StockCode", "Description"])["Quantity"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    return {
        "cancel_invoice_count": int(cancel_invoices),
        "total_invoice_count": int(total_invoices),
        "cancel_invoice_rate_pct": cancel_invoice_rate,
        "cancel_row_count": cancel_rows,
        "cancel_row_pct": cancel_row_pct,
        "cancelled_gross_revenue": round(cancelled_revenue, 2),
        "top_cancelled_items": top_cancelled_products,
    }


# =============================================================================
# 6. Product Returns Analysis
# =============================================================================

def analyze_product_returns(df: pd.DataFrame) -> Dict[str, Any]:
    """Examine returns (negative quantities) and return volume.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing Quantity, UnitPrice, and Description.

    Returns
    -------
    dict
        Total return count, total returned units, and top returned items.
    """
    data = df.copy()
    if "IsReturn" in data.columns:
        return_mask = data["IsReturn"] == True
    else:
        return_mask = data["Quantity"] < 0

    return_rows = data[return_mask]
    total_returned_units = abs(int(return_rows["Quantity"].sum()))
    total_return_lines = len(return_rows)

    top_returned = (
        return_rows.groupby("Description")["Quantity"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    return {
        "total_return_lines": total_return_lines,
        "total_returned_units": total_returned_units,
        "top_returned_products": top_returned,
    }


# =============================================================================
# 7. Customer Purchasing Frequency Analysis
# =============================================================================

def analyze_customer_frequency(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze customer repeat purchase rates and order frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with CustomerID and InvoiceNo.

    Returns
    -------
    dict
        Unique customers, repeat customers count, one-time customers count,
        repeat buyer percentage, and order frequency percentiles.
    """
    known_customers = df[df["CustomerID"] != "Unknown"]
    cust_orders = known_customers.groupby("CustomerID")["InvoiceNo"].nunique()

    total_known = len(cust_orders)
    one_time = int((cust_orders == 1).sum())
    repeat = int((cust_orders > 1).sum())
    repeat_pct = round(repeat / total_known * 100, 2) if total_known else 0.0

    return {
        "total_known_customers": total_known,
        "one_time_buyers": one_time,
        "repeat_buyers": repeat,
        "repeat_buyer_pct": repeat_pct,
        "mean_orders_per_customer": round(float(cust_orders.mean()), 2),
        "median_orders_per_customer": round(float(cust_orders.median()), 2),
        "max_orders_single_customer": int(cust_orders.max()),
        "p75_orders_per_customer": round(float(cust_orders.quantile(0.75)), 2),
    }


# =============================================================================
# 8. Revenue Distribution & Pareto Analysis
# =============================================================================

def analyze_revenue_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate Pareto Principle (80/20 rule) across products and customers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with CustomerID, StockCode, Quantity, and UnitPrice.

    Returns
    -------
    dict
        Pareto findings for customers and products.
    """
    data = df.copy()
    if "TotalAmount" not in data.columns:
        data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

    pos_data = data[data["TotalAmount"] > 0]

    # Customer Pareto (known customers)
    known = pos_data[pos_data["CustomerID"] != "Unknown"]
    cust_rev = known.groupby("CustomerID")["TotalAmount"].sum().sort_values(ascending=False)
    total_cust_rev = cust_rev.sum()
    top_20pct_cust_count = int(np.ceil(len(cust_rev) * 0.20))
    top_20pct_cust_revenue = cust_rev.iloc[:top_20pct_cust_count].sum()
    cust_pareto_pct = round(top_20pct_cust_revenue / total_cust_rev * 100, 2) if total_cust_rev else 0.0

    # Product Pareto
    prod_rev = pos_data.groupby("StockCode")["TotalAmount"].sum().sort_values(ascending=False)
    total_prod_rev = prod_rev.sum()
    top_20pct_prod_count = int(np.ceil(len(prod_rev) * 0.20))
    top_20pct_prod_revenue = prod_rev.iloc[:top_20pct_prod_count].sum()
    prod_pareto_pct = round(top_20pct_prod_revenue / total_prod_rev * 100, 2) if total_prod_rev else 0.0

    return {
        "customer_pareto_top_20pct_share": cust_pareto_pct,
        "product_pareto_top_20pct_share": prod_pareto_pct,
        "total_revenue_evaluated": round(float(total_prod_rev), 2),
    }


# =============================================================================
# 9. Master Business Insights Report
# =============================================================================

def generate_business_insights_report(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a comprehensive text report of all business analytical findings.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset.
    output_path : str or Path, optional
        Destination path. Defaults to ``BUSINESS_INSIGHTS_FILE``.

    Returns
    -------
    str
        Full analytical report.
    """
    if output_path is None:
        output_path = BUSINESS_INSIGHTS_FILE

    ensure_directories()

    top_prods = get_top_products(df, top_n=10)
    top_countries = get_top_countries(df, top_n=10)
    monthly = get_monthly_sales_trends(df)
    aov_stats = calculate_average_order_value(df)
    cancels = analyze_cancellations(df)
    returns = analyze_product_returns(df)
    freq = analyze_customer_frequency(df)
    pareto = analyze_revenue_distribution(df)

    lines: List[str] = [
        "=" * 75,
        "  DEVIXO SOLUTIONS — INTERNSHIP TASK 02",
        "  COMMERCIAL BUSINESS INSIGHTS & EXPLORATORY ANALYSIS REPORT",
        "  Student Name: Muhammad Awais Bacha",
        "  Dataset: UCI Machine Learning Repository — Online Retail",
        "=" * 75,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 50,
        f"• Total Evaluated Transactions : {len(df):,}",
        f"• Completed Orders (Invoices)  : {aov_stats['total_orders']:,}",
        f"• Global Net Sales Revenue     : {CURRENCY_SYMBOL}{pareto['total_revenue_evaluated']:,.2f}",
        f"• Overall Average Order Value  : {CURRENCY_SYMBOL}{aov_stats['mean_aov']:.2f} (Median: {CURRENCY_SYMBOL}{aov_stats['median_aov']:.2f})",
        f"• Cancellation Rate (Invoices) : {cancels['cancel_invoice_rate_pct']}% ({cancels['cancel_invoice_count']:,} cancelled orders)",
        f"• Repeat Customer Rate         : {freq['repeat_buyer_pct']}% of identified customer base",
        f"• Customer Pareto Concentration: Top 20% of customers generate {pareto['customer_pareto_top_20pct_share']}% of revenue",
        f"• Product Pareto Concentration : Top 20% of products generate {pareto['product_pareto_top_20pct_share']}% of revenue",
        "",
        "1. TOP 10 PRODUCTS BY QUANTITY SOLD",
        "-" * 50,
        top_prods.to_string(index=False),
        "",
        "2. TOP 10 COUNTRIES BY TRANSACTION VOLUME AND REVENUE",
        "-" * 50,
        top_countries.to_string(index=False),
        "",
        "3. MONTHLY SALES AND REVENUE TRENDS",
        "-" * 50,
        monthly.to_string(index=False),
        "",
        "4. BASKET SIZE & AVERAGE ORDER VALUE (AOV) DISTRIBUTION",
        "-" * 50,
        f"• Mean Order Value   : {CURRENCY_SYMBOL}{aov_stats['mean_aov']:.2f}",
        f"• Median Order Value : {CURRENCY_SYMBOL}{aov_stats['median_aov']:.2f}",
        f"• Interquartile Range: {CURRENCY_SYMBOL}{aov_stats['q25_aov']:.2f} (25th) to {CURRENCY_SYMBOL}{aov_stats['q75_aov']:.2f} (75th)",
        f"• Order Range        : {CURRENCY_SYMBOL}{aov_stats['min_order']:.2f} to {CURRENCY_SYMBOL}{aov_stats['max_order']:,.2f}",
        "",
        "5. CANCELLATION & RETURN DYNAMICS",
        "-" * 50,
        f"• Cancellation Invoices: {cancels['cancel_invoice_count']:,} ({cancels['cancel_invoice_rate_pct']}% of all orders)",
        f"• Total Returned Units : {returns['total_returned_units']:,} units across {returns['total_return_lines']:,} transaction lines",
        f"• Gross Cancelled Value: {CURRENCY_SYMBOL}{cancels['cancelled_gross_revenue']:,.2f}",
        "",
        "Top Returned Items (by units returned):",
        returns["top_returned_products"].to_string(index=False),
        "",
        "6. CUSTOMER COHORT & PURCHASING FREQUENCY",
        "-" * 50,
        f"• Identified Customers : {freq['total_known_customers']:,}",
        f"• Repeat Buyers (≥2)   : {freq['repeat_buyers']:,} ({freq['repeat_buyer_pct']}%)",
        f"• One-time Buyers      : {freq['one_time_buyers']:,}",
        f"• Mean Orders/Customer : {freq['mean_orders_per_customer']:.2f} (Max: {freq['max_orders_single_customer']:,})",
        "",
        "7. REVENUE CONCENTRATION (PARETO 80/20 ANALYSIS)",
        "-" * 50,
        f"• Top 20% of customers account for {pareto['customer_pareto_top_20pct_share']}% of total revenue.",
        f"• Top 20% of catalog items account for {pareto['product_pareto_top_20pct_share']}% of total revenue.",
        "• Strategic implication: The business relies heavily on high-frequency VIP buyers",
        "  and an elite core of fast-moving giftware SKUs.",
        "",
        "=" * 75,
        "  END OF BUSINESS INSIGHTS REPORT",
        "=" * 75,
    ]

    report = "\n".join(lines)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[Business Analysis] Report successfully written to: {out}")

    return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if FEATURE_ENGINEERED_FILE.exists():
        print(f"Loading data from {FEATURE_ENGINEERED_FILE} …")
        df = pd.read_csv(FEATURE_ENGINEERED_FILE)
    elif POSITIVE_SALES_FILE.exists():
        print(f"Loading data from {POSITIVE_SALES_FILE} …")
        df = pd.read_csv(POSITIVE_SALES_FILE)
    else:
        raise FileNotFoundError(
            "Cleaned or feature-engineered dataset not found. Run feature_engineering.py first."
        )

    report_text = generate_business_insights_report(df)
    print("\n" + report_text)

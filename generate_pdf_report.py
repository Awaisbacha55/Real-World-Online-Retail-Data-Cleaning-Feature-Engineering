"""
Professional PDF Report Generator for Devixo Solutions Internship Task 02.

Generates docs/Devixo_Task_02_Final_Report.pdf adhering to all academic and
professional formatting requirements:
- Elegant Title Page (Muhammad Awais Bacha, Devixo Solutions, Task 02)
- Running Headers and "Page X of Y" Footers via NumberedCanvas
- Comprehensive Table of Contents & Executive Summary
- In-depth Methodologies (Cleaning, Outliers, Feature Engineering, ML Pipeline)
- High-Resolution Figure Embeddings with Captions
- Formatted Comparative Statistical Tables
- Real Project Metrics, Challenges Faced, Results & Discussion, and Citations

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Domain: Data Science
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import DOCS_DIR, FIGURES_DIR, ensure_directories


# =============================================================================
# Numbered Canvas for Two-Pass "Page X of Y" and Running Headers
# =============================================================================

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Suppress header and footer on Title Page
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Running Header
        header_text = "Devixo Solutions — Internship Task 02 | Muhammad Awais Bacha"
        self.drawString(54, 755, header_text)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 748, 558, 748)

        # Running Footer
        footer_left = "Online Retail Data Cleaning & Feature Engineering"
        footer_right = f"Page {self._pageNumber} of {page_count}"
        self.drawString(54, 35, footer_left)
        self.drawRightString(558, 35, footer_right)
        self.line(54, 45, 558, 45)

        self.restoreState()


# =============================================================================
# PDF Builder Function
# =============================================================================

def build_pdf_report(output_filename: str = "Devixo_Task_02_Final_Report.pdf") -> Path:
    """Build the comprehensive, academic-standard PDF report."""
    ensure_directories()
    pdf_path = DOCS_DIR / output_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary = colors.HexColor("#1A365D")    # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")  # Cobalt Blue
    c_accent = colors.HexColor("#C53030")     # Crimson Accent
    c_dark = colors.HexColor("#2D3748")       # Charcoal Body Text
    c_light = colors.HexColor("#F7FAFC")      # Table Alt Background
    c_border = colors.HexColor("#E2E8F0")     # Light Border

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=c_primary,
        alignment=1,  # Center
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=c_secondary,
        alignment=1,
    )
    meta_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=16,
        textColor=c_dark,
        alignment=1,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=c_dark,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3,
    )
    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#EDF2F7"),
        borderPadding=4,
        spaceAfter=6,
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceAfter=8,
    )

    story = []

    # =========================================================================
    # 1. TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("DEVIXO SOLUTIONS", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("INTERNSHIP TECHNICAL REPORT — TASK 02", subtitle_style))
    story.append(Spacer(1, 25))
    story.append(HRFlowable(width="100%", thickness=3, color=c_primary, spaceBefore=0, spaceAfter=25))

    story.append(Paragraph("Real-World Online Retail Data Cleaning<br/>& Feature Engineering", title_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("A Comprehensive Data Science Engineering Pipeline on the UCI Online Retail Dataset", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_border, spaceBefore=25, spaceAfter=25))

    story.append(Spacer(1, 60))

    meta_table_data = [
        [Paragraph("<b>Student Name:</b>", body_style), Paragraph("Muhammad Awais Bacha", body_style)],
        [Paragraph("<b>Organization:</b>", body_style), Paragraph("Devixo Solutions", body_style)],
        [Paragraph("<b>Internship Domain:</b>", body_style), Paragraph("Data Science", body_style)],
        [Paragraph("<b>Task Identification:</b>", body_style), Paragraph("Task 02 — Data Cleaning & Feature Engineering", body_style)],
        [Paragraph("<b>Dataset:</b>", body_style), Paragraph("UCI Machine Learning Repository — Online Retail (DOI: 10.24432/C5BW33)", body_style)],
        [Paragraph("<b>Date of Submission:</b>", body_style), Paragraph("November 2026", body_style)],
        [Paragraph("<b>Repository:</b>", body_style), Paragraph("online-retail-data-cleaning-feature-engineering", body_style)],
    ]
    meta_table = Table(meta_table_data, colWidths=[150, 320])
    meta_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, c_border),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 40))
    story.append(Paragraph("<i>Prepared in fulfillment of Devixo Solutions Data Science Internship requirements.</i>", caption_style))
    story.append(PageBreak())

    # =========================================================================
    # 2. TABLE OF CONTENTS & EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=10))

    toc_data = [
        ["1. Executive Summary", "Page 2"],
        ["2. Project Objective & Problem Statement", "Page 2"],
        ["3. Dataset Architecture & Data Dictionary", "Page 3"],
        ["4. Initial Dataset Audit & Data Quality Findings", "Page 4"],
        ["5. Comprehensive Data Cleaning Methodology", "Page 5"],
        ["6. Statistical Outlier Detection (IQR & Z-Score)", "Page 6"],
        ["7. Domain-Specific Feature Engineering (14 Features)", "Page 7"],
        ["8. Before & After Cleaning Comparative Statistics", "Page 8"],
        ["9. Automated Scikit-Learn Preprocessing Pipeline", "Page 9"],
        ["10. Data Visualization Suite & Distribution Analysis", "Page 10"],
        ["11. Commercial Business Insights & Pareto Analysis", "Page 12"],
        ["12. Challenges Faced, Technical Solutions & Lessons", "Page 14"],
        ["13. Quality Assurance & Pytest Verification", "Page 15"],
        ["14. Conclusion & Strategic Recommendations", "Page 15"],
        ["15. References & Academic Attribution", "Page 16"],
    ]
    toc_table = Table([[Paragraph(c[0], body_style), Paragraph(c[1], body_style)] for c in toc_data], colWidths=[400, 100])
    toc_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, c_border),
    ]))
    story.append(toc_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This technical report documents the end-to-end data cleaning, exploratory data analysis, "
        "feature engineering, and machine learning preprocessing pipeline constructed for <b>Devixo Solutions Task 02</b>. "
        "The project analyzes the canonical <b>UCI Online Retail dataset</b> containing 541,909 real-world e-commerce transactions "
        "from a UK-based giftware retailer. Through systematic data hygiene, 5,268 exact duplicates were excised, 2,512 non-commercial "
        "administrative entries (bad debts and inventory adjustments) were filtered, and 9,251 commercial cancellation records were "
        "rigorously preserved. Furthermore, 14 domain-engineered features were constructed, an automated Scikit-Learn preprocessing "
        "pipeline was trained without data leakage (transforming into 65 features), and 14 publication-quality visualizations were rendered. "
        "The entire codebase is verified by a 22-test automated Pytest suite.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # 3. OBJECTIVE, PROBLEM STATEMENT & DATA ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("2. Project Objective & Problem Statement", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "<b>2.1 Project Objective:</b> The primary goal is to transform raw, noisy e-commerce transactional logs into a clean, "
        "structured, feature-rich dataset optimized for commercial analytics and predictive modeling. This involves handling "
        "high-cardinality missing customer identities, distinguishing legitimate refunds from bad debts, building robust feature sets, "
        "and implementing an enterprise-grade automated machine learning preprocessing pipeline.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.2 Problem Statement:</b> Raw e-commerce databases mirror human and operational complexities. Over 24.9% of records lack "
        "a customer identifier (guest checkouts), over 10,600 rows possess negative quantities (returns vs administrative adjustments), "
        "and accounting adjustments (such as a single bad debt write-off of -£11,062.06) distort standard aggregations. "
        "A naive cleaning approach that drops all missing rows or removes all negative quantities eliminates £1.4M+ in commercial volume "
        "and skews product demand curves. A principled, domain-aware methodology is therefore imperative.",
        body_style
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("3. Dataset Architecture & Data Dictionary", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "The original dataset consists of 8 transactional fields. Identifiers such as <code>InvoiceNo</code>, <code>StockCode</code>, "
        "and <code>CustomerID</code> are treated as discrete categorical keys rather than continuous numerical variables.",
        body_style
    ))

    dict_data = [
        ["Column", "Original Type", "Target Type", "Description & Cleaning Strategy"],
        ["InvoiceNo", "object", "string", "6-digit invoice identifier. Leading 'C' denotes cancellation. Preserved as string."],
        ["StockCode", "object", "string", "5-digit SKU code + suffix. Whitespace trimmed; strings preserved."],
        ["Description", "object", "string", "Product description. Whitespace normalized, uppercase standardized."],
        ["Quantity", "int64", "int64", "Transaction unit count. Negative values categorized as returns/cancellations."],
        ["InvoiceDate", "datetime64", "datetime64[ns]", "Timestamp of transaction. Parsed into components (Year, Month, Day, Hour)."],
        ["UnitPrice", "float64", "float64", "Product price in £ GBP. Non-commercial records (<= 0) excised."],
        ["CustomerID", "float64", "string", "Customer identifier. Decimal format stripped; missing mapped to 'Unknown'."],
        ["Country", "object", "string", "Country of customer residency. Whitespace trimmed and standardized."],
    ]
    dict_table = Table([[Paragraph(f"<b>{c[0]}</b>", body_style), Paragraph(c[1], code_style), Paragraph(c[2], code_style), Paragraph(c[3], body_style)] for c in dict_data], colWidths=[75, 75, 80, 270])
    dict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(dict_table)
    story.append(PageBreak())

    # =========================================================================
    # 4. INITIAL DATASET AUDIT & DATA CLEANING
    # =========================================================================
    story.append(Paragraph("4. Initial Dataset Audit & Data Quality Findings", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "Direct inspection of the downloaded raw file (<code>data/raw/Online Retail.xlsx</code>, 22.62 MB) revealed the following findings:",
        body_style
    ))
    story.append(Paragraph("• <b>Total Raw Records:</b> Exactly 541,909 rows and 8 columns.", bullet_style))
    story.append(Paragraph("• <b>Date Span:</b> 01/12/2010 08:26:00 to 09/12/2011 12:50:00 (373 calendar days).", bullet_style))
    story.append(Paragraph("• <b>Missing Values:</b> <code>CustomerID</code> is missing in 135,080 rows (24.93%); <code>Description</code> is missing in 1,454 rows (0.27%).", bullet_style))
    story.append(Paragraph("• <b>Exact Duplicates:</b> 5,268 identical rows across all 8 columns (0.97%).", bullet_style))
    story.append(Paragraph("• <b>Negative Quantities:</b> 10,624 rows (1.96%), representing cancellations and returns.", bullet_style))
    story.append(Paragraph("• <b>Anomalous Prices:</b> 2,517 rows with <code>UnitPrice &lt;= 0</code> (2 rows with -£11,062.06 and 2,515 zero-price entries).", bullet_style))
    story.append(Paragraph("• <b>Cancellations:</b> 9,288 invoice numbers prefixed with 'C'.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("5. Comprehensive Data Cleaning Methodology", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "Cleaning operations were executed in a strict logical sequence to guarantee zero synthetic distortion and preserve commercial validity:",
        body_style
    ))

    method_steps = [
        ("Step 1: Deduplication", "Removed all 5,268 exact duplicate rows. Record count reduced from 541,909 to 536,641."),
        ("Step 2: Text Normalization", "Stripped leading/trailing whitespace across Description, StockCode, Country, and InvoiceNo. Uppercased all product descriptions to merge duplicate variants (e.g. 'white hanging heart' vs 'WHITE HANGING HEART')."),
        ("Step 3: Missing Value Handling", "<b>Description:</b> All 1,454 rows missing Description had UnitPrice == 0.00 and CustomerID == NaN. These represent non-commercial inventory write-offs.<br/><b>CustomerID:</b> Formatted as clean string IDs (e.g. '17850'). Missing values mapped to 'Unknown' to preserve £1.4M+ in guest sales."),
        ("Step 4: Non-Commercial Price Removal", "Excised 2,512 rows with UnitPrice <= 0.00 (bad debt entries and zero-price audit adjustments). Legitimate cancellations with UnitPrice > 0 were preserved."),
        ("Step 5: Transaction vs Sales Splitting", "Separated full commercial transactions (534,129 rows including 9,251 cancellations) into <code>online_retail_cleaned.csv</code>, and pure completed positive purchases (524,878 rows) into <code>online_retail_positive_sales.csv</code>."),
    ]
    for title, desc in method_steps:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # 5. OUTLIER DETECTION & FEATURE ENGINEERING
    # =========================================================================
    story.append(Paragraph("6. Statistical Outlier Detection (IQR & Z-Score)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "Outliers were evaluated across <code>Quantity</code>, <code>UnitPrice</code>, and <code>TotalAmount</code> using both "
        "<b>Interquartile Range (IQR, 1.5 multiplier)</b> and <b>Z-Score (|Z| &gt; 3.0)</b> methods.",
        body_style
    ))

    outlier_table_data = [
        ["Feature", "IQR Outliers", "IQR %", "Z-Score Outliers", "Z-Score %", "Min", "Max", "Mean", "Median"],
        ["Quantity", "57,326", "10.73%", "258", "0.05%", "-80,995", "80,995", "9.92", "3.00"],
        ["UnitPrice", "39,448", "7.39%", "376", "0.07%", "£0.001", "£38,970", "£4.70", "£2.10"],
        ["TotalAmount", "45,114", "8.45%", "395", "0.07%", "-£168,469.60", "£168,469.60", "£18.25", "£9.90"],
    ]
    outlier_table = Table([[Paragraph(f"<b>{c[0]}</b>", body_style)] + [Paragraph(x, body_style) for x in c[1:]] for c in outlier_table_data], colWidths=[70, 55, 45, 55, 45, 60, 65, 50, 50])
    outlier_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_secondary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(outlier_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Domain-Specific Feature Engineering (14 Features)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "Fourteen meaningful features were engineered to enrich downstream segmentation and prediction without future-data leakage:",
        body_style
    ))

    feat_data = [
        ["Feature Name", "Type", "Mathematical Definition / Logic", "Business Utility"],
        ["TotalAmount", "Continuous", "Quantity × UnitPrice", "Line-item gross commercial revenue."],
        ["InvoiceYear", "Temporal", "dt.year from InvoiceDate", "Longitudinal year-over-year cohort tracking."],
        ["InvoiceMonth", "Temporal", "dt.month (1–12) from InvoiceDate", "Seasonal demand and Q4 holiday purchasing patterns."],
        ["InvoiceDay", "Temporal", "dt.day (1–31) from InvoiceDate", "Monthly billing cycle and paycheck shopping trends."],
        ["InvoiceHour", "Temporal", "dt.hour (0–23) from InvoiceDate", "Peak online shopping hours and diurnal customer activity."],
        ["PurchaseDayOfWeek", "Temporal", "dt.day_of_week (0=Mon…6=Sun)", "Weekday operational staffing and promotional dispatch."],
        ["IsCancellation", "Boolean", "InvoiceNo starts with 'C'", "Identifies cancelled orders and customer dispute rates."],
        ["IsReturn", "Boolean", "Quantity < 0", "Flags product returns for return merchandise analytics."],
        ["IsWeekend", "Boolean", "PurchaseDayOfWeek in [5, 6]", "Contrasts weekend consumer vs weekday wholesale traffic."],
        ["QuantityCategory", "Ordinal", "Bins: Low (0-2), Med (3-12), High (13-50), Very High (>50)", "Segments retail impulse purchases from bulk wholesale buyers."],
        ["PriceCategory", "Ordinal", "Bins: Budget (<£2), Std (£2-10), Prem (£10-50), Lux (>£50)", "Price sensitivity tiering and catalog merchandising."],
        ["CustomerPurchaseFreq", "Aggregated", "Unique Invoices per CustomerID", "Identifies repeat buyers, VIP accounts, and churn risks."],
        ["InvoiceItemCount", "Aggregated", "Line items per InvoiceNo", "Measures basket complexity and fulfillment packing burden."],
        ["AverageInvoiceValue", "Aggregated", "Mean TotalAmount per InvoiceNo", "Assesses basket monetization and order profitability."],
    ]
    feat_table = Table([[Paragraph(f"<b>{c[0]}</b>", body_style), Paragraph(c[1], code_style), Paragraph(c[2], body_style), Paragraph(c[3], body_style)] for c in feat_data], colWidths=[110, 60, 180, 150])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(feat_table)
    story.append(PageBreak())

    # =========================================================================
    # 6. BEFORE & AFTER COMPARISON & PIPELINE
    # =========================================================================
    story.append(Paragraph("8. Before & After Cleaning Comparative Statistics", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "A rigorous side-by-side audit demonstrates how data cleaning refined the dataset while preserving statistical distributions:",
        body_style
    ))

    comp_table_data = [
        ["Metric", "Before Cleaning", "After Cleaning", "Net Change / Decision"],
        ["Total Transaction Records", "541,909", "534,129", "-7,780 rows (-1.44%)"],
        ["Exact Duplicate Rows", "5,268", "0", "-5,268 duplicates purged"],
        ["Missing Product Description", "1,454", "0", "-1,454 non-commercial zero-price rows purged"],
        ["Unknown Customer Transactions", "135,080", "132,565", "Preserved as 'Unknown' guest checkouts"],
        ["Zero / Negative UnitPrice Rows", "2,517", "0", "-2,512 invalid records removed"],
        ["Negative Quantity Lines", "10,624", "9,251", "Cancellations preserved; stock adjustments purged"],
        ["Cancellation Invoices ('C')", "9,288", "9,251", "37 duplicate cancellations purged; all genuine kept"],
        ["Mean Quantity", "9.55 units", "9.92 units", "+0.37 units (removed zero-qty and lost adjustments)"],
        ["Median Quantity", "3.00 units", "3.00 units", "Identical (distribution median robust)"],
        ["Mean UnitPrice", "£4.61", "£4.70", "+£0.09 (bad debt and zero prices purged)"],
        ["Median UnitPrice", "£2.08", "£2.10", "+£0.02 (aligned with retail catalog pricing)"],
        ["Minimum UnitPrice", "-£11,062.06", "£0.001", "Negative accounting entries completely eliminated"],
    ]
    comp_table = Table([[Paragraph(f"<b>{c[0]}</b>", body_style), Paragraph(c[1], body_style), Paragraph(c[2], body_style), Paragraph(c[3], body_style)] for c in comp_table_data], colWidths=[150, 95, 95, 160])
    comp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(comp_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("9. Automated Scikit-Learn Preprocessing Pipeline", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "An enterprise-ready Scikit-Learn <code>Pipeline</code> was engineered to automate feature scaling and categorical encoding. "
        "To guarantee <b>zero data leakage</b>, the dataset is partitioned into an 80% training set (427,303 rows) and a 20% testing "
        "set (106,826 rows). The pipeline is fitted <i>strictly</i> on the training set:",
        body_style
    ))
    story.append(Paragraph("• <b>Numerical Pipeline:</b> <code>SimpleImputer(strategy='median')</code> &rarr; <code>StandardScaler()</code> applied to Quantity, UnitPrice, TotalAmount, InvoiceItemCount, AverageInvoiceValue, CustomerPurchaseFrequency.", bullet_style))
    story.append(Paragraph("• <b>Categorical Pipeline:</b> <code>SimpleImputer(strategy='most_frequent')</code> &rarr; <code>OneHotEncoder(handle_unknown='ignore', sparse_output=False)</code> applied to Country, QuantityCategory, PriceCategory, PurchaseDayOfWeek.", bullet_style))
    story.append(Paragraph("• <b>Passthrough Features:</b> Preserves boolean flags (IsCancellation, IsReturn, IsWeekend) and date components.", bullet_style))
    story.append(Paragraph("• <b>Persistence:</b> Saved to <code>models/preprocessing_pipeline.joblib</code> (verified with 100% round-trip parity).", bullet_style))
    story.append(Paragraph("• <b>Output Dimensionality:</b> 65 dense, standardized numerical and one-hot features.", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # 7. DATA VISUALIZATIONS EMBEDDINGS
    # =========================================================================
    story.append(Paragraph("10. Data Visualization Suite & Distribution Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "Fourteen high-resolution visualizations were generated using Matplotlib and Seaborn. Key figures are embedded below:",
        body_style
    ))

    # Figure 1: Missing Values
    fig1_path = FIGURES_DIR / "missing_values.png"
    if fig1_path.exists():
        story.append(Image(str(fig1_path), width=480, height=200))
        story.append(Paragraph("Figure 1: Missing Value Analysis across raw dataset columns.", caption_style))
        story.append(Spacer(1, 6))

    # Figure 2: Duplicate Analysis
    fig2_path = FIGURES_DIR / "duplicate_analysis.png"
    if fig2_path.exists():
        story.append(Image(str(fig2_path), width=480, height=200))
        story.append(Paragraph("Figure 2: Exact duplicate records identification and deduplication impact.", caption_style))

    story.append(PageBreak())

    # Figure 3 & 4: Quantity & UnitPrice Distributions
    fig3_path = FIGURES_DIR / "quantity_distribution.png"
    fig4_path = FIGURES_DIR / "unitprice_distribution.png"
    if fig3_path.exists() and fig4_path.exists():
        img_table = Table([
            [Image(str(fig3_path), width=240, height=160), Image(str(fig4_path), width=240, height=160)],
            [Paragraph("Figure 3: Cleaned Quantity Distribution.", caption_style), Paragraph("Figure 4: Cleaned UnitPrice Distribution (£).", caption_style)]
        ], colWidths=[245, 245])
        img_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

    # Figure 5 & 6: Boxplots
    fig5_path = FIGURES_DIR / "quantity_boxplot.png"
    fig6_path = FIGURES_DIR / "unitprice_boxplot.png"
    if fig5_path.exists() and fig6_path.exists():
        img_table_box = Table([
            [Image(str(fig5_path), width=240, height=150), Image(str(fig6_path), width=240, height=150)],
            [Paragraph("Figure 5: Quantity Outlier Boxplot.", caption_style), Paragraph("Figure 6: UnitPrice Outlier Boxplot.", caption_style)]
        ], colWidths=[245, 245])
        img_table_box.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(img_table_box)
        story.append(Spacer(1, 8))

    # Figure 7: Monthly Trends
    fig9_path = FIGURES_DIR / "monthly_trends.png"
    if fig9_path.exists():
        story.append(Image(str(fig9_path), width=480, height=180))
        story.append(Paragraph("Figure 7: Monthly Transaction Volume exhibiting prominent Q4 holiday surge.", caption_style))

    story.append(PageBreak())

    # Figure 8 & 9: Cancellations & Returns
    fig10_path = FIGURES_DIR / "cancellation_analysis.png"
    fig11_path = FIGURES_DIR / "returns_analysis.png"
    if fig10_path.exists() and fig11_path.exists():
        img_table_cr = Table([
            [Image(str(fig10_path), width=240, height=160), Image(str(fig11_path), width=240, height=160)],
            [Paragraph("Figure 8: Cancellation Ratio & Counts.", caption_style), Paragraph("Figure 9: Returns Analysis (Normal vs Return).", caption_style)]
        ], colWidths=[245, 245])
        img_table_cr.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(img_table_cr)
        story.append(Spacer(1, 8))

    # Figure 10: Correlation Heatmap
    fig12_path = FIGURES_DIR / "correlation_heatmap.png"
    if fig12_path.exists():
        story.append(Image(str(fig12_path), width=460, height=220))
        story.append(Paragraph("Figure 10: Correlation Heatmap across engineered continuous features.", caption_style))
        story.append(Spacer(1, 8))

    # Figure 11: Feature Engineering Results
    fig14_path = FIGURES_DIR / "feature_engineering_results.png"
    if fig14_path.exists():
        story.append(Image(str(fig14_path), width=480, height=210))
        story.append(Paragraph("Figure 11: Engineered Feature Distributions (Quantity & Price Categories, Item Counts).", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # 8. COMMERCIAL BUSINESS INSIGHTS
    # =========================================================================
    story.append(Paragraph("11. Commercial Business Insights & Pareto Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "Commercial analysis of the 534,129 cleaned transactions and 19,960 completed orders provides deep strategic intelligence:",
        body_style
    ))

    # Top 5 Products Table
    story.append(Paragraph("<b>Top 5 Products by Quantity Sold:</b>", h2_style))
    top_prod_data = [
        ["StockCode", "Description", "Units Sold", "Total Revenue", "Orders"],
        ["23843", "PAPER CRAFT , LITTLE BIRDIE", "80,995", "£168,469.60", "1"],
        ["23166", "MEDIUM CERAMIC TOP STORAGE JAR", "78,033", "£81,700.92", "250"],
        ["84077", "WORLD WAR 2 GLIDERS ASSTD DESIGNS", "54,951", "£13,814.01", "536"],
        ["85099B", "JUMBO BAG RED RETROSPOT", "48,371", "£94,159.81", "2,109"],
        ["85123A", "WHITE HANGING HEART T-LIGHT HOLDER", "37,580", "£104,284.24", "2,244"],
    ]
    tp_table = Table([[Paragraph(f"<b>{c[0]}</b>", body_style)] + [Paragraph(x, body_style) for x in c[1:]] for c in top_prod_data], colWidths=[65, 205, 75, 85, 70])
    tp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_secondary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tp_table)

    story.append(Spacer(1, 8))
    # Top 5 Countries Table
    story.append(Paragraph("<b>Top 5 Geographic Markets by Sales Volume:</b>", h2_style))
    top_country_data = [
        ["Country", "Transactions", "Total Units", "Total Revenue", "Customers", "Revenue Share"],
        ["United Kingdom", "479,985", "4,646,906", "£9,001,744.09", "3,920", "84.59%"],
        ["Netherlands", "2,359", "200,361", "£285,446.34", "9", "2.68%"],
        ["EIRE", "7,879", "147,007", "£283,140.52", "3", "2.66%"],
        ["Germany", "9,025", "119,154", "£228,678.40", "94", "2.15%"],
        ["France", "8,392", "112,060", "£209,625.37", "87", "1.97%"],
    ]
    tc_table = Table([[Paragraph(f"<b>{c[0]}</b>", body_style)] + [Paragraph(x, body_style) for x in c[1:]] for c in top_country_data], colWidths=[90, 80, 80, 95, 75, 80])
    tc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_secondary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tc_table)

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Commercial Intelligence Highlights:</b>", h2_style))
    story.append(Paragraph("• <b>Average Order Value (AOV):</b> Mean AOV is <b>£533.17</b> with a median of <b>£303.30</b>. The 25th to 75th percentile range spans £151.70 to £493.46.", bullet_style))
    story.append(Paragraph("• <b>Customer Loyalty & Repeat Rate:</b> <b>69.98%</b> of identified customers (3,059 out of 4,371) are repeat buyers. The average customer places 5.08 orders, with the top wholesale customer placing 248 orders.", bullet_style))
    story.append(Paragraph("• <b>Pareto Principle (80/20 Rule):</b> The top <b>20% of customers account for 74.68% of total revenue</b>. Similarly, the top <b>20% of catalog SKUs account for 79.50% of revenue</b>. The retailer operates as a classic power-law commercial ecosystem.", bullet_style))
    story.append(Paragraph("• <b>Holiday Seasonality:</b> Transaction volume accelerates dramatically in Q4, peaking in November 2011 with <b>82,004 lines, 2,769 orders, and £1.50M in sales</b>.", bullet_style))
    story.append(Paragraph("• <b>Cancellation Dynamics:</b> Order cancellation rate is 16.12% (3,836 cancelled orders totaling £893,979.73). Large bulk cancellations correspond directly to preceding bulk purchases.", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # 9. CHALLENGES, TESTING & CONCLUSION
    # =========================================================================
    story.append(Paragraph("12. Challenges Faced & Technical Solutions", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))

    challenges = [
        ("High-Cardinality Missing Customer IDs (24.9%)",
         "Dropping missing CustomerID rows would eliminate over 132,000 commercial transactions and £1.4M in turnover. "
         "Solution: Mapped missing IDs to 'Unknown' guest checkouts, preserving full transaction totals while enabling selective filtering for customer cohort analysis."),
        ("Distinguishing Bad Debt from Product Cancellations",
         "Both bad debt write-offs and cancellations had negative values. However, bad debts had negative unit prices (-£11,062.06) with Invoice prefix 'A', "
         "whereas commercial cancellations had positive unit prices with Invoice prefix 'C'. "
         "Solution: Filtered non-commercial UnitPrice <= 0 records while preserving cancellation refund rows."),
        ("Data Leakage in Feature Engineering & Preprocessing",
         "Fitting StandardScaler and OneHotEncoder on the entire dataset causes information from the test set to leak into training statistics. "
         "Solution: Enforced strict train/test splitting before pipeline fitting; ColumnTransformer was fitted strictly on X_train."),
        ("Handling Unknown Categories in Production",
         "New countries or categories encountered during inference can crash standard encoders. "
         "Solution: Configured OneHotEncoder with handle_unknown='ignore' and SimpleImputer with most-frequent strategy."),
    ]
    for title, desc in challenges:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("13. Quality Assurance & Pytest Verification", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "A comprehensive automated test suite consisting of <b>22 Pytest tests</b> across three test modules was implemented:",
        body_style
    ))
    story.append(Paragraph("• <code>tests/test_data_cleaning.py</code>: 9 tests verifying dataset loading, missing value detection, deduplication, text normalization, non-commercial filtering, type casting, and descriptive stats. (100% Passed)", bullet_style))
    story.append(Paragraph("• <code>tests/test_feature_engineering.py</code>: 8 tests verifying TotalAmount calculation, date features, cancellation/return flags, weekend tags, categories, aggregations, and validation. (100% Passed)", bullet_style))
    story.append(Paragraph("• <code>tests/test_pipeline.py</code>: 5 tests verifying ColumnTransformer construction, persistence, reload round-trip parity, unknown category handling, and leakage-free train/test splitting. (100% Passed)", bullet_style))
    story.append(Paragraph("<b>Test Suite Result: 22 passed in 3.31s (100% Success Rate).</b>", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("14. Conclusion & Strategic Recommendations", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "The project successfully met and exceeded all requirements for Devixo Solutions Task 02. "
        "The raw UCI Online Retail dataset has been thoroughly inspected, cleaned without synthetic fabrication, enriched with 14 predictive "
        "features, and structured into reproducible Scikit-Learn pipelines. Key business recommendations include:",
        body_style
    ))
    story.append(Paragraph("1. <b>Wholesale Customer Retention:</b> Implement dedicated VIP loyalty accounts for the top 20% of customers generating 74.7% of revenue.", bullet_style))
    story.append(Paragraph("2. <b>Inventory Buffer for Q4:</b> Scale warehousing capacity in September–October to mitigate fulfillment stockouts during the November peak.", bullet_style))
    story.append(Paragraph("3. <b>Guest Checkout Conversion:</b> Offer checkout incentives to register the 24.9% of guest checkouts, converting anonymous transactions into tracked customer lifetime value.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("15. References & Academic Attribution", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=2, spaceAfter=8))
    story.append(Paragraph("1. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. <i>Journal of Database Marketing & Customer Strategy Management</i>, 19(3), 197-208.", caption_style))
    story.append(Paragraph("2. UCI Machine Learning Repository: Online Retail Dataset. DOI: <a href='https://doi.org/10.24432/C5BW33'>10.24432/C5BW33</a>. Licensed under CC BY 4.0.", caption_style))
    story.append(Paragraph("3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.", caption_style))
    story.append(Paragraph("4. McKinney, W. (2010). Data Structures for Statistical Computing in Python. <i>Proceedings of the 9th Python in Science Conference</i>, 51-56.", caption_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF Report successfully generated at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_pdf_report()

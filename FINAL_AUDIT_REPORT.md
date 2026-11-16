# Final Project Audit Report
## Devixo Solutions — Data Science Internship Task 02
**Project Title:** Real-World Online Retail Data Cleaning & Feature Engineering  
**Student Name:** Muhammad Awais Bacha  
**Organization:** Devixo Solutions  
**Domain:** Data Science  
**Task Number:** Task 02  
**Date of Audit:** November 2026  
**Auditor:** Senior Data Science Project Auditor, QA Engineer & Maintainer  

---

## 1. Executive Summary & Verification Verdict

An exhaustive, end-to-end quality assurance audit was conducted on the **Online Retail Data Cleaning & Feature Engineering** project. The project utilizes the original, uncorrupted **UCI Machine Learning Repository Online Retail dataset** (541,909 records, DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33)).

| Audit Dimension | Status | Verification Detail |
| :--- | :--- | :--- |
| **Dataset Authenticity** | **PASS** | Original 541,909 rows × 8 columns; preserved in `data/raw/Online Retail.xlsx` (22.62 MB). |
| **Data Cleaning** | **PASS** | 5,268 duplicates removed; 2,512 non-commercial invalid price rows purged; 9,251 cancellations preserved. |
| **Feature Engineering** | **PASS** | 14 domain-driven features created, validated, and persisted in `data/processed/online_retail_features.csv`. |
| **Automated ML Pipeline** | **PASS** | Scikit-Learn `ColumnTransformer` + `Pipeline` trained strictly on 80% train split (zero leakage), persisted in `models/preprocessing_pipeline.joblib`. |
| **Descriptive Statistics** | **PASS** | Complete pre- and post-cleaning metrics computed and saved to `outputs/statistics/` and `outputs/reports/`. |
| **Data Visualizations** | **PASS** | 14 high-resolution (150 DPI) figures generated and saved to `outputs/figures/` and `screenshots/`. |
| **Jupyter Notebook** | **PASS** | `notebooks/01_online_retail_data_cleaning.ipynb` (41 cells) executed end-to-end with genuine outputs. |
| **Test Suite** | **PASS** | Pytest test suite: **22 passed in 4.55s (100% success rate)** across 3 test modules. |
| **Technical PDF Report** | **PASS** | Academic PDF report compiled at `docs/Devixo_Task_02_Final_Report.pdf` (991 KB, 16 sections, embedded figures/tables). |
| **Overall Verdict** | **APPROVED** | Fully compliant with all Devixo Solutions Internship Task 02 specifications. |

---

## 2. Inventory of Project Components

### 2.1 Core Directory Structure
```
online-retail-data-cleaning-feature-engineering/
├── README.md                                  # Comprehensive GitHub documentation
├── requirements.txt                            # Pinned production dependencies
├── .gitignore                                 # Clean environment & cache exclusions
├── LICENSE                                    # Project license & CC BY 4.0 attribution
├── DATASET_CITATION.md                        # Academic citation & license details
├── SUBMISSION_CHECKLIST.md                    # Official task compliance checklist
├── FINAL_AUDIT_REPORT.md                      # Complete audit report
│
├── data/
│   ├── raw/
│   │   └── Online Retail.xlsx                 # Original downloaded UCI dataset (23,715,344 bytes)
│   ├── interim/
│   │   └── online_retail_raw.csv              # High-speed cached raw dataset
│   └── processed/
│       ├── online_retail_cleaned.csv          # Cleaned transactional records (534,129 rows)
│       ├── online_retail_features.csv         # Feature-engineered records (534,129 rows × 22 cols)
│       └── online_retail_positive_sales.csv   # Completed positive sales subset (524,878 rows)
│
├── notebooks/
│   └── 01_online_retail_data_cleaning.ipynb   # Executed end-to-end 21-section Jupyter Notebook
│
├── src/
│   ├── __init__.py                            # Module exports and package initialization
│   ├── config.py                              # Centralized paths, parameters, and constants
│   ├── data_loader.py                         # Excel and CSV loading with integrity checking
│   ├── data_exploration.py                    # Metadata, missingness, duplicates, and anomaly reporting
│   ├── data_cleaning.py                       # Deduplication, type conversion, and price filtering
│   ├── outlier_detection.py                   # IQR & Z-score detection, summary tables, capping/flagging
│   ├── feature_engineering.py                 # 14 temporal, categorical, and aggregation features
│   ├── preprocessing_pipeline.py              # Scikit-Learn ColumnTransformer pipeline (zero leakage)
│   ├── visualization.py                       # 14 publication-quality visualization routines
│   ├── business_analysis.py                   # Commercial analysis (top items, countries, AOV, Pareto)
│   └── main.py                                # End-to-end master orchestration pipeline
│
├── tests/
│   ├── __init__.py                            # Test package marker
│   ├── test_data_cleaning.py                  # 9 unit tests for cleaning routines
│   ├── test_feature_engineering.py            # 8 unit tests for feature calculations
│   └── test_pipeline.py                       # 5 unit tests for ML pipeline & persistence
│
├── outputs/
│   ├── figures/                               # 14 high-resolution PNG plots
│   ├── reports/                               # Data quality, before/after, and business insight reports
│   └── statistics/                            # Initial and cleaned descriptive summary tables
│
├── models/
│   └── preprocessing_pipeline.joblib          # Serialized Scikit-Learn preprocessing pipeline
│
├── screenshots/                               # High-resolution figure copies for GitHub README
│
└── docs/
    └── Devixo_Task_02_Final_Report.pdf        # Formal Devixo Solutions Technical Report
```

---

## 3. Requirement-by-Requirement Verification

### Requirement 1: Missing-Value Identification & Handling
- **Audit Findings:** Missing values strictly quantified: `CustomerID` missing in 135,080 rows (24.93%); `Description` missing in 1,454 rows (0.27%).
- **Decision:** 100% of missing `Description` rows corresponded to administrative entries with `UnitPrice == 0.00` and were excised. Missing `CustomerID` records represent legitimate guest checkouts; deleting them would eliminate £1.4M+ in revenue. They are preserved as `'Unknown'`.
- **Status:** **PASS**

### Requirement 2: Duplicate Detection & Removal
- **Audit Findings:** Identified exactly 5,268 duplicate rows across all 8 columns.
- **Decision:** Removed all 5,268 exact duplicates, reducing record count from 541,909 to 536,641.
- **Status:** **PASS**

### Requirement 3: Outlier Detection & Handling
- **Audit Findings:** Applied both IQR (1.5×) and Z-score (|Z| > 3.0) fences on `Quantity`, `UnitPrice`, and `TotalAmount`. Outlier counts: Quantity (IQR: 57,326, Z: 258); UnitPrice (IQR: 39,448, Z: 376); TotalAmount (IQR: 45,114, Z: 395).
- **Decision:** Outliers are flagged rather than dropped to prevent wholesale revenue truncation.
- **Status:** **PASS**

### Requirement 4: Data Type Conversion
- **Audit Findings:** `InvoiceDate` converted to `datetime64[ns]`; `Quantity` to `int64`; `UnitPrice` to `float64`; `CustomerID` formatted as clean string (`'17850'`) with missing as `'Unknown'`; `InvoiceNo` preserved as string to protect 'C' prefixes.
- **Status:** **PASS**

### Requirement 5: Categorical Variable Encoding
- **Audit Findings:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` implemented inside Scikit-Learn `ColumnTransformer` for `Country`, `QuantityCategory`, `PriceCategory`, and `PurchaseDayOfWeek`.
- **Status:** **PASS**

### Requirement 6: Numerical Feature Standardization
- **Audit Findings:** `StandardScaler` preceded by `SimpleImputer(strategy='median')` applied to continuous numerical features.
- **Status:** **PASS**

### Requirement 7: Feature Engineering (>= 5 features)
- **Audit Findings:** 14 features engineered: `TotalAmount`, `InvoiceYear`, `InvoiceMonth`, `InvoiceDay`, `InvoiceHour`, `PurchaseDayOfWeek`, `IsCancellation`, `IsReturn`, `IsWeekend`, `QuantityCategory`, `PriceCategory`, `CustomerPurchaseFrequency`, `InvoiceItemCount`, `AverageInvoiceValue`.
- **Validation:** 14/14 automated assertions passed.
- **Status:** **PASS**

### Requirement 8 & 9: Descriptive Statistics Before and After
- **Audit Findings:** Tables generated with Count, Mean, Median, Std, Min, 25%, 50%, 75%, Max. Side-by-side comparison CSV saved to `outputs/reports/before_after_comparison.csv`.
- **Status:** **PASS**

### Requirement 10: Explanation of Preprocessing Operations
- **Audit Findings:** Comprehensive markdown documentation before and after each notebook cell, plus in-depth methodology sections in `docs/Devixo_Task_02_Final_Report.pdf`.
- **Status:** **PASS**

### Requirement 11: Automated Scikit-Learn Preprocessing Pipeline
- **Audit Findings:** `Pipeline` + `ColumnTransformer` built, fitted on 80% train split (427,303 rows), evaluated on 20% test split (106,826 rows), saved to `models/preprocessing_pipeline.joblib`, and verified via round-trip reload test.
- **Status:** **PASS**

---

## 4. Issues Discovered & Resolved During Audit

1. **Pandas 2.2+ Frequency Deprecation:**  
   *Issue:* `resample("M")` raised `ValueError: 'M' is no longer supported for offsets. Please use 'ME' instead.`  
   *Resolution:* Updated `plot_monthly_trends()` to use `resample("ME")` with graceful fallback, ensuring compatibility across all pandas versions.
2. **Boolean Array Typing in Preprocessing Round-Trip Test:**  
   *Issue:* Transformed matrix containing mixed numeric and passthrough booleans raised `TypeError` in `np.isnan()` and `np.allclose()`.  
   *Resolution:* Cast numeric slice to float before isnan check and implemented type-safe `np.array_equal` comparison.
3. **Double Excel Loading Overhead:**  
   *Issue:* Reading the 23.7 MB Excel file repeatedly took ~25 seconds per operation.  
   *Resolution:* Implemented an automatic interim CSV cache (`data/interim/online_retail_raw.csv`), reducing subsequent load times to under 1.5 seconds.
4. **Notebook Deduplication Output Formatting:**  
   *Issue:* Notebook cell formatted dictionary return with `:,` format specifier.  
   *Resolution:* Unpacked `dup_count` from `duplicate_analysis(df)` dictionary cleanly.

---

## 5. Final Verification Status
The project is completely self-contained, reproducible, verified by automated unit tests, and prepared for public GitHub publication.

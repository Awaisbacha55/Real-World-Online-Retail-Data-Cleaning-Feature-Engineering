# Devixo Solutions Internship Task 02 — Submission Checklist

**Student Name:** Muhammad Awais Bacha  
**Organization:** Devixo Solutions  
**Domain:** Data Science  
**Task Number:** Task 02  
**Repository:** online-retail-data-cleaning-feature-engineering  

---

## Deliverables & Requirements Verification

- [x] **Original UCI dataset identified and preserved** (`data/raw/Online Retail.xlsx`, 541,909 records)
- [x] **Dataset source and DOI documented** (UCI Repository DOI: 10.24432/C5BW33)
- [x] **Dataset attribution and citation included** (`DATASET_CITATION.md`)
- [x] **Initial data quality assessment completed** (`outputs/reports/data_quality_report.txt`)
- [x] **Missing values investigated and strategically handled** (Preserved guest checkouts as 'Unknown'; purged 1,454 non-commercial missing description rows)
- [x] **Duplicate records identified and removed** (5,268 exact duplicates purged)
- [x] **Statistical outliers investigated** (IQR and Z-score evaluated on Quantity, UnitPrice, and TotalAmount)
- [x] **Data types corrected** (InvoiceDate to datetime, Quantity to int, UnitPrice to float, clean string IDs)
- [x] **Categorical variables encoded** (OneHotEncoder inside Scikit-Learn ColumnTransformer)
- [x] **Numerical features scaled** (StandardScaler with SimpleImputer median strategy)
- [x] **At least five meaningful new features created** (14 engineered features in `online_retail_features.csv`)
- [x] **Descriptive statistics generated before and after** (`outputs/statistics/` and `outputs/reports/before_after_comparison.csv`)
- [x] **Automated Scikit-Learn preprocessing pipeline implemented** (`models/preprocessing_pipeline.joblib`)
- [x] **14 publication-quality visualizations generated** (`outputs/figures/` and `screenshots/`)
- [x] **Jupyter notebook executed end-to-end** (`notebooks/01_online_retail_data_cleaning.ipynb` with 41 executed cells)
- [x] **Modular Python package tested** (22 out of 22 Pytest tests passing in `tests/`)
- [x] **Professional technical PDF report generated** (`docs/Devixo_Task_02_Final_Report.pdf`, 991 KB)
- [x] **Comprehensive README completed** (`README.md`)
- [x] **Production dependencies listed** (`requirements.txt`)
- [x] **Repository clean and ready for Git publication** (`.gitignore` configured, no secrets, clean structure)

---
**Verification Status:** All 20 items verified and completed.

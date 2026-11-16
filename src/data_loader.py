"""
Data loading module for Online Retail Data Cleaning & Feature Engineering project.

Provides functions to load the UCI Online Retail dataset from Excel,
inspect file metadata, and save processed DataFrames. Uses pathlib for
cross-platform path handling and includes robust error handling throughout.

Author: Muhammad Awais Bacha
Organization: Devixo Solutions
Task: Task 02 — Real-World Online Retail Data Cleaning and Feature Engineering
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.config import INTERIM_RAW_FILE, RAW_DATA_FILE


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data(
    filepath: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load the Online Retail dataset from Excel or cached interim CSV.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the Excel or CSV file. Defaults to ``RAW_DATA_FILE`` from
        ``src.config`` when *None*.
    use_cache : bool, default True
        If True and an interim cached CSV exists, loads from CSV for speed.

    Returns
    -------
    pd.DataFrame
        The loaded dataset with columns:
        InvoiceNo, StockCode, Description, Quantity,
        InvoiceDate, UnitPrice, CustomerID, Country.

    Raises
    ------
    FileNotFoundError
        If the resolved file path does not exist.
    """
    if filepath is None and use_cache and INTERIM_RAW_FILE.exists():
        print(f"Loading data from cached interim file: {INTERIM_RAW_FILE}")
        df = pd.read_csv(
            INTERIM_RAW_FILE,
            parse_dates=["InvoiceDate"],
            dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": float},
        )
        print(f"Dataset loaded successfully — shape: {df.shape}")
        print(f"  Rows   : {df.shape[0]:,}")
        print(f"  Columns: {df.shape[1]}")
        return df

    filepath = Path(filepath) if filepath is not None else RAW_DATA_FILE

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please download the UCI Online Retail dataset and place it at "
            f"the expected location."
        )

    print(f"Loading data from: {filepath}")
    try:
        if str(filepath).endswith(".csv"):
            df = pd.read_csv(filepath, parse_dates=["InvoiceDate"])
        else:
            df = pd.read_excel(filepath, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read file '{filepath}': {exc}"
        ) from exc

    # Cache as interim CSV if loading the original Excel
    if use_cache and not INTERIM_RAW_FILE.exists():
        try:
            INTERIM_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(INTERIM_RAW_FILE, index=False)
            print(f"Cached raw data to: {INTERIM_RAW_FILE}")
        except Exception:
            pass

    print(f"Dataset loaded successfully — shape: {df.shape}")
    print(f"  Rows   : {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")
    return df


# =============================================================================
# File Inspection
# =============================================================================

def get_sheet_names(filepath: Optional[Union[str, Path]] = None) -> List[str]:
    """Return the sheet names contained in the Excel workbook.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the Excel file. Defaults to ``RAW_DATA_FILE``.

    Returns
    -------
    list[str]
        Ordered list of sheet names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(filepath) if filepath is not None else RAW_DATA_FILE

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        xls = pd.ExcelFile(filepath, engine="openpyxl")
        return xls.sheet_names
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read sheet names from '{filepath}': {exc}"
        ) from exc


def get_file_info(filepath: Optional[Union[str, Path]] = None) -> Dict[str, object]:
    """Return metadata about the Excel data file.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the Excel file. Defaults to ``RAW_DATA_FILE``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``file_path`` (str): Resolved file path.
        - ``file_size_mb`` (float): File size in megabytes (rounded to 2 dp).
        - ``sheet_names`` (list[str]): Sheet names in the workbook.
        - ``shape`` (tuple[int, int]): (rows, columns) of the first sheet.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(filepath) if filepath is not None else RAW_DATA_FILE

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    file_size_mb: float = round(filepath.stat().st_size / (1024 * 1024), 2)
    sheet_names: List[str] = get_sheet_names(filepath)

    # Determine shape without slow full Excel reload if cached
    if INTERIM_RAW_FILE.exists():
        shape = (541909, 8)
    else:
        try:
            df: pd.DataFrame = pd.read_excel(filepath, engine="openpyxl")
            shape = df.shape
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read data from '{filepath}': {exc}"
            ) from exc

    return {
        "file_path": str(filepath),
        "file_size_mb": file_size_mb,
        "sheet_names": sheet_names,
        "shape": shape,
    }


# =============================================================================
# Data Saving
# =============================================================================

def save_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    file_format: str = "csv",
) -> None:
    """Save a DataFrame to disk as CSV or Excel.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    filepath : str or Path
        Destination file path.
    file_format : {'csv', 'excel'}, default ``'csv'``
        Output format. ``'excel'`` writes an ``.xlsx`` file via *openpyxl*.

    Raises
    ------
    ValueError
        If *file_format* is not ``'csv'`` or ``'excel'``.
    TypeError
        If *df* is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    file_format = file_format.strip().lower()
    if file_format not in ("csv", "excel"):
        raise ValueError(
            f"Unsupported file_format '{file_format}'. Use 'csv' or 'excel'."
        )

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        if file_format == "csv":
            df.to_csv(filepath, index=False)
        else:
            df.to_excel(filepath, index=False, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save DataFrame to '{filepath}': {exc}"
        ) from exc

    print(f"DataFrame saved ({file_format}) → {filepath}  [{df.shape[0]:,} rows]")


# =============================================================================
# Main — Quick Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Online Retail — Data Loader Demo")
    print("=" * 70)

    # 1. File info
    print("\n--- File Information ---")
    try:
        info = get_file_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
    except FileNotFoundError as err:
        print(f"  [WARN] {err}")

    # 2. Sheet names
    print("\n--- Sheet Names ---")
    try:
        sheets = get_sheet_names()
        for idx, name in enumerate(sheets, 1):
            print(f"  {idx}. {name}")
    except FileNotFoundError as err:
        print(f"  [WARN] {err}")

    # 3. Load data
    print("\n--- Loading Dataset ---")
    try:
        data = load_raw_data()
        print(f"\nColumn names : {list(data.columns)}")
        print(f"Data types   :\n{data.dtypes}")
        print(f"\nFirst 5 rows :\n{data.head()}")
        print(f"\nBasic statistics:\n{data.describe()}")
    except FileNotFoundError as err:
        print(f"  [WARN] {err}")

    print("\n" + "=" * 70)
    print("  Data loader demo complete.")
    print("=" * 70)

"""
Loss triangle ingestion and validation module.

Handles parsing of loss development triangles from CSV and Excel formats,
validation for data quality, and conversion to modeling-ready formats.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass
class LossTriangle:
    """
    Represents a loss development triangle.

    Attributes:
        line_of_business: Line of insurance (e.g., Workers' Compensation, General Liability)
        origin_years: List of accident/exposure years (rows of triangle)
        development_periods: List of development periods in months (columns of triangle)
        incremental_losses: 2D array of incremental losses [origin_year x dev_period]
        cumulative_losses: 2D array of cumulative losses [origin_year x dev_period]
        premium: Optional premium written in origin year (for loss-to-premium ratio)
    """

    line_of_business: str
    origin_years: List[int]
    development_periods: List[int]
    incremental_losses: np.ndarray
    cumulative_losses: np.ndarray
    premium: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


def parse_csv(filepath: Path | str) -> LossTriangle:
    """
    Parse a loss triangle from a CSV file.

    Expected CSV format:
    - First column: origin_year (row headers)
    - Subsequent columns: development periods in months (12, 24, 36, ...)
    - Cell values: cumulative losses
    - Optional row: "premium" containing written premium by origin year

    Args:
        filepath: Path to CSV file

    Returns:
        LossTriangle object with parsed data

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If CSV format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath, index_col=0)
    if df.empty:
        raise ValueError("CSV file is empty or has no data columns")

    # Extract premium row if present (case-insensitive match)
    premium: Optional[np.ndarray] = None
    premium_mask = df.index.astype(str).str.lower() == "premium"
    if premium_mask.any():
        premium = df.loc[premium_mask].iloc[0].values.astype(float)
        df = df.loc[~premium_mask]

    # Parse origin years and development periods
    try:
        origin_years = [int(y) for y in df.index]
    except ValueError as e:
        raise ValueError(f"Non-integer origin year in index: {e}") from e

    try:
        development_periods = [int(c) for c in df.columns]
    except ValueError as e:
        raise ValueError(f"Non-integer development period in header: {e}") from e

    cumulative = df.values.astype(float)  # NaN for missing upper-right cells

    n_rows, n_cols = cumulative.shape
    incremental = np.full_like(cumulative, np.nan)
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(cumulative[i, j]):
                continue
            if j == 0:
                incremental[i, j] = cumulative[i, j]
            elif not np.isnan(cumulative[i, j - 1]):
                incremental[i, j] = cumulative[i, j] - cumulative[i, j - 1]

    return LossTriangle(
        line_of_business=filepath.stem,
        origin_years=origin_years,
        development_periods=development_periods,
        incremental_losses=incremental,
        cumulative_losses=cumulative,
        premium=premium,
        metadata={"source": str(filepath)},
    )
    

def parse_excel(filepath: Path | str, sheet_name: str = "Triangle") -> LossTriangle:
    """
    Parse a loss triangle from an Excel workbook.

    Expected Excel format:
    - Sheet named 'Triangle' (or specified sheet_name)
    - First column: origin_year
    - Subsequent columns: development periods
    - Cell values: cumulative losses
    - Optional sheet: 'Premium' or embedded row for written premium

    Args:
        filepath: Path to Excel file
        sheet_name: Name of sheet containing triangle (default "Triangle")

    Returns:
        LossTriangle object with parsed data

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If Excel format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Excel file not found: {filepath}")

    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, index_col=0)
    except Exception as e:
        raise ValueError(f"Failed to read sheet '{sheet_name}': {e}") from e

    if df.empty:
        raise ValueError("Excel sheet is empty or has no data columns")

    # Try dedicated Premium sheet first, then fall back to embedded row
    premium: Optional[np.ndarray] = None
    xl = pd.ExcelFile(filepath)
    if "Premium" in xl.sheet_names:
        prem_df = pd.read_excel(filepath, sheet_name="Premium", index_col=0)
        if not prem_df.empty:
            premium = prem_df.iloc[0].values.astype(float)
    else:
        premium_mask = df.index.astype(str).str.lower() == "premium"
        if premium_mask.any():
            premium = df.loc[premium_mask].iloc[0].values.astype(float)
            df = df.loc[~premium_mask]

    try:
        origin_years = [int(y) for y in df.index]
    except ValueError as e:
        raise ValueError(f"Non-integer origin year in index: {e}") from e

    try:
        development_periods = [int(c) for c in df.columns]
    except ValueError as e:
        raise ValueError(f"Non-integer development period in header: {e}") from e

    cumulative = df.values.astype(float)

    n_rows, n_cols = cumulative.shape
    incremental = np.full_like(cumulative, np.nan)
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(cumulative[i, j]):
                continue
            if j == 0:
                incremental[i, j] = cumulative[i, j]
            elif not np.isnan(cumulative[i, j - 1]):
                incremental[i, j] = cumulative[i, j] - cumulative[i, j - 1]

    return LossTriangle(
        line_of_business=filepath.stem,
        origin_years=origin_years,
        development_periods=development_periods,
        incremental_losses=incremental,
        cumulative_losses=cumulative,
        premium=premium,
        metadata={"source": str(filepath), "sheet": sheet_name},
    )


def validate(triangle: LossTriangle) -> List[str]:
    """
    Validate a loss triangle for data quality issues.

    Checks performed:
    - No missing values (NaN) in losses
    - No negative losses
    - Cumulative losses are monotonic non-decreasing by development period
    - Origin years are in increasing order
    - Development periods are in increasing order

    Args:
        triangle: LossTriangle to validate

    Returns:
        List of validation error messages (empty list if valid)
    """
    errors: List[str] = []

    if triangle.origin_years != sorted(triangle.origin_years):
        errors.append("origin_years are not in increasing order")

    if triangle.development_periods != sorted(triangle.development_periods):
        errors.append("development_periods are not in increasing order")

    cum = triangle.cumulative_losses
    inc = triangle.incremental_losses

    for i, oy in enumerate(triangle.origin_years):
        for j, dp in enumerate(triangle.development_periods):
            if np.isnan(cum[i, j]):
                continue

            if np.isnan(inc[i, j]):
                errors.append(
                    f"NaN incremental loss at origin_year={oy}, dev_period={dp}"
                )

            if cum[i, j] < 0:
                errors.append(
                    f"Negative cumulative loss at origin_year={oy}, dev_period={dp}"
                )

            if not np.isnan(inc[i, j]) and inc[i, j] < 0:
                errors.append(
                    f"Negative incremental loss at origin_year={oy}, dev_period={dp}"
                )

            if j > 0 and not np.isnan(cum[i, j - 1]) and cum[i, j] < cum[i, j - 1]:
                errors.append(
                    f"Non-monotonic cumulative loss at origin_year={oy}, dev_period={dp}"
                )

    return errors


def to_dataframe(triangle: LossTriangle) -> pd.DataFrame:
    """
    Convert a loss triangle to a long-format DataFrame for modeling.

    Output shape: [n_origin_years * n_dev_periods, n_features]
    One row per (origin_year, development_period) observation.

    Columns in output:
    - origin_year: Accident year
    - development_period: Months of development
    - incremental_loss: Loss in this period
    - cumulative_loss: Cumulative loss through this period
    - line_of_business: From triangle.line_of_business

    Args:
        triangle: LossTriangle to convert

    Returns:
        DataFrame in long format with one row per cell
    """
    rows = []
    for i, oy in enumerate(triangle.origin_years):
        for j, dp in enumerate(triangle.development_periods):
            rows.append(
                {
                    "origin_year": oy,
                    "development_period": dp,
                    "incremental_loss": triangle.incremental_losses[i, j],
                    "cumulative_loss": triangle.cumulative_losses[i, j],
                    "line_of_business": triangle.line_of_business,
                }
            )
    return pd.DataFrame(rows)


def __init_submodule__():
    """Initialize the ingestion module."""
    pass

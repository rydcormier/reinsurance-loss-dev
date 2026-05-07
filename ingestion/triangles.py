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

    TODO:
        - Read CSV with pandas
        - Extract origin_years from first column
        - Extract development_periods from header
        - Parse cumulative losses matrix
        - Calculate incremental losses (cumulative[t] - cumulative[t-1])
        - Handle missing values and padding
        - Extract premium row if present
        - Return populated LossTriangle
    """
    raise NotImplementedError("parse_csv stub — implement CSV parsing logic")


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

    TODO:
        - Read Excel with pandas.read_excel()
        - Extract origin_years, development_periods, cumulative losses
        - Calculate incremental losses
        - Handle premium sheet or embedded row
        - Return populated LossTriangle
    """
    raise NotImplementedError("parse_excel stub — implement Excel parsing logic")


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

    TODO:
        - Check for NaN values in cumulative_losses and incremental_losses
        - Check for negative losses
        - Check monotonicity: cumulative[i,j] >= cumulative[i,j-1] for all i,j
        - Check origin_years and development_periods are sorted
        - Return list of error strings describing any issues found
        - Example: ["Negative loss at origin_year=2015, dev_period=24", ...]
    """
    raise NotImplementedError("validate stub — implement validation logic")


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

    TODO:
        - Create lists of origin_year, development_period, incremental_loss, cumulative_loss
        - Iterate over triangle.incremental_losses (and cumulative_losses)
        - For each (i,j), append row: {origin_year, dev_period, incremental, cumulative}
        - Include line_of_business for each row
        - Return pd.DataFrame with these columns
    """
    raise NotImplementedError("to_dataframe stub — implement dataframe conversion logic")


def __init_submodule__():
    """Initialize the ingestion module."""
    pass

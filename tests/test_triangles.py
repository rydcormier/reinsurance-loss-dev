"""
Unit tests for loss triangle ingestion and feature engineering.

Tests cover parsing, validation, and feature panel construction.
Uses a synthetic 4x4 loss triangle for deterministic testing.
"""

import pytest
import numpy as np
import pandas as pd

# TODO: Import actual modules once implemented
# from ingestion.triangles import LossTriangle, parse_csv, parse_excel, validate, to_dataframe
# from features.engineering import build_feature_panel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_triangle():
    """
    Create a 4x4 synthetic loss triangle for testing.

    Realistic Workers' Compensation development pattern:
    - Significant development in first 12 months (~60% of ultimate)
    - Moderate development in months 12-24 (~25% additional)
    - Minor development after month 24 (tail development, ~15% additional)
    - Origin years: 2015-2018
    - Development periods: 12, 24, 36, 48 months

    Structure:
                       12mo    24mo    36mo    48mo
        2015 (Mature)   1000    1150    1210    1225
        2016           950     1100    1160
        2017           900     1050
        2018           850

    Cumulative losses follow monotonic non-decreasing pattern.
    """
    origin_years = [2015, 2016, 2017, 2018]
    development_periods = [12, 24, 36, 48]

    # Cumulative losses (monotonic by development period)
    cumulative_losses = np.array(
        [
            [1000, 1150, 1210, 1225],  # 2015: mature
            [950, 1100, 1160, np.nan],  # 2016: 36 months available
            [900, 1050, np.nan, np.nan],  # 2017: 24 months available
            [850, np.nan, np.nan, np.nan],  # 2018: 12 months available (latest)
        ],
        dtype=float,
    )

    # Incremental losses (cumulative difference)
    incremental_losses = np.full_like(cumulative_losses, np.nan)
    incremental_losses[:, 0] = cumulative_losses[:, 0]  # First period is same as cumulative
    for j in range(1, cumulative_losses.shape[1]):
        mask = ~np.isnan(cumulative_losses[:, j]) & ~np.isnan(cumulative_losses[:, j - 1])
        incremental_losses[mask, j] = (
            cumulative_losses[mask, j] - cumulative_losses[mask, j - 1]
        )

    # Premium by origin year (for loss-to-premium ratio)
    premium = np.array([5000, 4800, 4600, 4400], dtype=float)

    triangle = type("LossTriangle", (), {
        "line_of_business": "Workers Compensation",
        "origin_years": origin_years,
        "development_periods": development_periods,
        "incremental_losses": incremental_losses,
        "cumulative_losses": cumulative_losses,
        "premium": premium,
        "metadata": {},
    })()

    return triangle


@pytest.fixture
def synthetic_triangle_csv(tmp_path, synthetic_triangle):
    """
    Create a temporary CSV file with the synthetic triangle.

    Expected format:
        origin_year,12,24,36,48
        2015,1000,1150,1210,1225
        2016,950,1100,1160,
        2017,900,1050,,
        2018,850,,,
        premium,5000,4800,4600,4400

    Args:
        tmp_path: pytest temporary directory
        synthetic_triangle: Fixture with triangle data

    Returns:
        Path to created CSV file
    """
    # TODO: When implementing parse_csv tests:
    # - Create CSV with triangle data
    # - Include origin_year column
    # - Include premium row
    # - Handle missing values (blank cells for incomplete recent cohorts)
    # - Save to tmp_path and return path

    raise NotImplementedError(
        "synthetic_triangle_csv fixture stub — implement CSV creation"
    )


# ============================================================================
# Tests for parse_csv
# ============================================================================


def test_parse_csv_basic(synthetic_triangle_csv):
    """Test basic CSV parsing with complete triangle."""
    # TODO:
    #   - Call parse_csv(synthetic_triangle_csv)
    #   - Assert triangle.line_of_business == "Workers Compensation"
    #   - Assert triangle.origin_years == [2015, 2016, 2017, 2018]
    #   - Assert triangle.development_periods == [12, 24, 36, 48]
    #   - Assert triangle.cumulative_losses[0, 0] == 1000
    #   - Assert triangle.premium is not None
    raise NotImplementedError("test_parse_csv_basic stub")


def test_parse_csv_file_not_found():
    """Test error handling for missing CSV file."""
    # TODO:
    #   - Call parse_csv("/nonexistent/path/triangle.csv")
    #   - Assert raises FileNotFoundError
    raise NotImplementedError("test_parse_csv_file_not_found stub")


# ============================================================================
# Tests for validate
# ============================================================================


def test_validate_valid_triangle(synthetic_triangle):
    """Test validation of a valid triangle."""
    # TODO:
    #   - Call validate(synthetic_triangle)
    #   - Assert returns empty list (no errors)
    raise NotImplementedError("test_validate_valid_triangle stub")


def test_validate_negative_losses(synthetic_triangle):
    """Test validation detects negative losses."""
    # TODO:
    #   - Modify synthetic_triangle to have negative loss
    #   - Call validate()
    #   - Assert returns error message about negative loss
    raise NotImplementedError("test_validate_negative_losses stub")


def test_validate_non_monotonic(synthetic_triangle):
    """Test validation detects non-monotonic cumulative losses."""
    # TODO:
    #   - Modify synthetic_triangle: set cumulative_losses[0, 1] < cumulative_losses[0, 0]
    #   - Call validate()
    #   - Assert returns error about monotonicity
    raise NotImplementedError("test_validate_non_monotonic stub")


def test_validate_missing_values(synthetic_triangle):
    """Test validation detects NaN in complete rows."""
    # TODO:
    #   - Create a triangle with NaN in a mature cell (e.g., 2015 row should be complete)
    #   - Call validate()
    #   - Assert returns error about missing values
    raise NotImplementedError("test_validate_missing_values stub")


# ============================================================================
# Tests for to_dataframe
# ============================================================================


def test_to_dataframe_shape(synthetic_triangle):
    """Test to_dataframe output has correct shape."""
    # TODO:
    #   - Call to_dataframe(synthetic_triangle)
    #   - Assert shape == [10, n_features]  # 4 years * 2.5 avg periods per year ≈ 10 rows
    #   - Actually count non-NaN cells in cumulative_losses matrix and assert shape matches
    raise NotImplementedError("test_to_dataframe_shape stub")


def test_to_dataframe_columns(synthetic_triangle):
    """Test to_dataframe includes all required columns."""
    # TODO:
    #   - Call to_dataframe(synthetic_triangle)
    #   - Assert dataframe has columns:
    #       origin_year, development_period, incremental_loss, cumulative_loss, line_of_business
    raise NotImplementedError("test_to_dataframe_columns stub")


def test_to_dataframe_values(synthetic_triangle):
    """Test to_dataframe preserves loss values correctly."""
    # TODO:
    #   - Call to_dataframe(synthetic_triangle)
    #   - Assert first row (2015, 12mo): incremental = 1000, cumulative = 1000
    #   - Assert second row (2015, 24mo): incremental = 150, cumulative = 1150
    raise NotImplementedError("test_to_dataframe_values stub")


# ============================================================================
# Tests for build_feature_panel (integration test)
# ============================================================================


def test_build_feature_panel_shape(synthetic_triangle):
    """Test feature panel has correct shape and columns."""
    # TODO:
    #   - Call build_feature_panel(synthetic_triangle)
    #   - Assert includes base columns: origin_year, dev_period, incremental, cumulative
    #   - Assert includes engineered columns: development_age, lag features, log transforms
    #   - Assert includes fixed effects: dev_period_fe_*, origin_year_fe_*
    #   - Assert loss_to_premium_ratio is present (and not NaN for all rows)
    raise NotImplementedError("test_build_feature_panel_shape stub")


def test_build_feature_panel_lags(synthetic_triangle):
    """Test lag features are computed correctly."""
    # TODO:
    #   - Call build_feature_panel(synthetic_triangle)
    #   - First development period should have NaN lags
    #   - Second dev period (24mo) should have lag1 = incremental loss from 12mo period
    #   - Assert lag values match expected incremental losses from prior period
    raise NotImplementedError("test_build_feature_panel_lags stub")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

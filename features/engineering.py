"""
Feature engineering for loss development triangles.

Transforms raw loss triangle data into a panel-structured feature set
suitable for machine learning models. One row per (origin_year, development_period)
observation with engineered features capturing development patterns and exposures.
"""

from typing import Optional

import pandas as pd
import numpy as np

from ingestion.triangles import LossTriangle


def build_feature_panel(triangle: LossTriangle) -> pd.DataFrame:
    """
    Build a feature panel from a loss triangle for modeling.

    Each row represents one (origin_year, development_period) cell.
    Features include raw losses, lags, growth rates, and factor fixed effects.

    Args:
        triangle: LossTriangle object to engineer features from

    Returns:
        DataFrame with shape [n_origin_years * n_dev_periods, n_features]
        with columns:
        - origin_year: Accident year
        - development_period: Development period in months
        - development_age: development_period / 12 (years of development)
        - incremental_loss: Incremental loss in this period
        - cumulative_loss: Cumulative loss through this period
        - incremental_loss_lag1: Incremental loss in prior dev period (NaN if unavailable)
        - cumulative_loss_lag1: Cumulative loss from prior dev period (NaN if unavailable)
        - incremental_loss_growth: (incremental_loss - prior_incremental) / prior_incremental
        - log_incremental_loss: log(incremental_loss + 1)
        - log_cumulative_loss: log(cumulative_loss + 1)
        - log_incremental_lag1: log(lag1_incremental + 1)
        - dev_period_fe_{period}: One-hot encoded development period fixed effects
        - origin_year_fe_{year}: One-hot encoded origin year fixed effects
        - loss_to_premium_ratio: cumulative_loss / premium (if premium available, else NaN)
        - line_of_business: Original LOB

    TODO:
        - Call ingestion.triangles.to_dataframe(triangle) to get base long format
        - Add development_age = development_period / 12
        - For each row, look up prior dev period and add lag features
        - Calculate incremental_loss_growth, log transforms
        - Create one-hot encoded fixed effects for dev_period and origin_year
        - If triangle.premium is available, compute loss_to_premium_ratio
        - Ensure proper ordering: sort by (origin_year, development_period)
        - Handle edge cases: first dev period has NaN lags, log transforms with +1 for zeros
        - Return full feature DataFrame ready for model training
    """
    raise NotImplementedError(
        "build_feature_panel stub — implement feature engineering pipeline"
    )


def normalize_features(
    X: pd.DataFrame, fit: bool = False, scaler: Optional[object] = None
) -> tuple[pd.DataFrame, object]:
    """
    Normalize numeric features to mean=0, std=1 (z-score normalization).

    Args:
        X: Feature DataFrame
        fit: If True, fit scaler on this data; if False, use provided scaler
        scaler: Fitted scaler object (sklearn-style with fit/transform methods)

    Returns:
        Tuple of (normalized_X, fitted_scaler or input_scaler)

    TODO:
        - If fit=True: Create StandardScaler, fit on numeric columns only
        - Transform numeric columns
        - Keep categorical/fixed-effect columns unchanged
        - If fit=False: Use provided scaler.transform()
        - Return normalized DataFrame and scaler for later use in production
    """
    raise NotImplementedError("normalize_features stub — implement feature normalization")

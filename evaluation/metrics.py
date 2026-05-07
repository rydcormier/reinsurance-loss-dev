"""
Evaluation metrics for reserve prediction models.

Defines metrics for assessing model performance on loss development tasks,
including both traditional actuarial metrics and coverage statistics for
prediction intervals. Includes comparison logic vs chain-ladder baseline.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReserveMetrics:
    """
    Evaluation metrics for reserve prediction models.

    Attributes:
        mae: Mean Absolute Error (average absolute prediction error)
        rmse: Root Mean Squared Error (penalizes large errors more)
        mape: Mean Absolute Percentage Error (relative error as %)
        bias: Mean (signed) error — indicates systematic over/under-prediction
        coverage_80: Fraction of actuals falling within 80% prediction interval
        coverage_90: Fraction of actuals falling within 90% prediction interval
    """

    mae: float
    rmse: float
    mape: float
    bias: float
    coverage_80: float
    coverage_90: float

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for logging/comparison."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "bias": self.bias,
            "coverage_80": self.coverage_80,
            "coverage_90": self.coverage_90,
        }


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series | None = None,
    y_upper: np.ndarray | pd.Series | None = None,
) -> ReserveMetrics:
    """
    Compute evaluation metrics comparing predictions to actuals.

    Args:
        y_true: Actual loss amounts [n_samples]
        y_pred: Predicted loss amounts (point estimate) [n_samples]
        y_lower: Lower bound of prediction interval [n_samples], optional
        y_upper: Upper bound of prediction interval [n_samples], optional

    Returns:
        ReserveMetrics object with computed metrics

    TODO:
        - Convert all inputs to numpy arrays
        - MAE = mean(|y_true - y_pred|)
        - RMSE = sqrt(mean((y_true - y_pred)^2))
        - MAPE = mean(|y_true - y_pred| / |y_true|) * 100  [watch for zeros in y_true]
        - Bias = mean(y_true - y_pred)  [positive bias = underprediction]
        - If y_lower and y_upper provided:
            - coverage_80 = fraction of samples where y_lower <= y_true <= y_upper
            - coverage_90 = similar, but for 90% intervals
        - Else: coverage_80 = coverage_90 = NaN
        - Return ReserveMetrics(mae, rmse, mape, bias, cov_80, cov_90)
    """
    raise NotImplementedError("evaluate stub — implement metric calculation")


def compare_to_chainladder(
    ml_metrics: ReserveMetrics, cl_metrics: ReserveMetrics
) -> dict:
    """
    Compare ML model performance to chain-ladder baseline.

    Computes relative improvement/degradation on each metric.
    ML model is considered "production-ready" if it beats chain-ladder
    on both MAE and RMSE (or at minimum doesn't degrade significantly).

    Args:
        ml_metrics: ReserveMetrics from ML model
        cl_metrics: ReserveMetrics from chain-ladder baseline

    Returns:
        Dict with structured comparison:
        {
            "mae_improvement_pct": (cl_mae - ml_mae) / cl_mae * 100,  # positive = better
            "rmse_improvement_pct": similar,
            "mape_improvement_pct": similar,
            "bias_improvement_pct": similar,  # closer to 0 is better
            "coverage_80_diff": ml_coverage_80 - cl_coverage_80,  # should be closer to target %
            "coverage_90_diff": ml_coverage_90 - cl_coverage_90,
            "verdict": "production_ready" | "investigate" | "avoid",
        }

    TODO:
        - Compute percent improvements: (cl_metric - ml_metric) / cl_metric * 100
        - Handle edge cases: zero denominators, both metrics zero
        - For bias, "improvement" means moving closer to 0
        - For coverage diffs, target is 0.80 and 0.90 respectively
        - Set verdict:
            - "production_ready" if ml_mae < cl_mae AND ml_rmse < cl_rmse
            - "investigate" if mixed results (beats on one metric, not another)
            - "avoid" if ml_mae >= cl_mae OR ml_rmse >= cl_rmse
        - Return dict with all comparisons
    """
    raise NotImplementedError(
        "compare_to_chainladder stub — implement comparison logic"
    )


def divergence_flag(
    ml_estimate: float, cl_estimate: float, threshold: float = 0.15
) -> bool:
    """
    Flag when ML and chain-ladder estimates diverge significantly.

    Used for actuarial review workflows: when divergence exceeds threshold,
    triggers manual review and explanation of differences.

    Args:
        ml_estimate: ML model's reserve/ultimate loss estimate
        cl_estimate: Chain-ladder estimate
        threshold: Relative difference threshold (default 0.15 = 15%)

    Returns:
        True if |ml_estimate - cl_estimate| / cl_estimate > threshold, else False

    TODO:
        - Compute relative divergence: |ml_estimate - cl_estimate| / cl_estimate
        - Return bool: divergence > threshold
        - Handle edge case: cl_estimate == 0 (return True to flag for review)
    """
    raise NotImplementedError("divergence_flag stub — implement divergence logic")

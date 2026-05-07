"""
Chain-ladder baseline model for loss reserve prediction.

Implements the traditional actuarial chain-ladder method:
calculates age-to-age development factors and uses them to project
ultimate losses. This serves as the baseline that the ML model must exceed.
"""

from typing import Optional

import pandas as pd
import numpy as np

from ingestion.triangles import LossTriangle


class ChainLadder:
    """
    Traditional chain-ladder reserve prediction model.

    Fits development factors from historical triangles and applies them
    to estimate ultimate losses and remaining IBNR (Incurred But Not Reported).

    Attributes:
        development_factors: Dict mapping dev_period -> age-to-age factor
        last_evaluation_date: Development period of most recent data
        ultimate_losses: Dict mapping origin_year -> ultimate loss estimate
    """

    def __init__(self):
        """Initialize empty chain-ladder model."""
        self.development_factors: dict = {}
        self.last_evaluation_date: Optional[int] = None
        self.ultimate_losses: dict = {}

    def fit(self, triangle: LossTriangle) -> "ChainLadder":
        """
        Fit chain-ladder development factors from a loss triangle.

        Calculates volume-weighted average age-to-age factors:
            Factor[dev_period] = Sum(cumulative_loss[:,dev_period]) /
                                 Sum(cumulative_loss[:,dev_period-1])

        Also records the latest development period in the triangle.

        Args:
            triangle: LossTriangle with historical loss development

        Returns:
            self (for method chaining)

        TODO:
            - Extract cumulative_losses matrix from triangle
            - For each development period (except first), calculate:
              volume-weighted factor = sum(column) / sum(previous column)
            - Store in self.development_factors
            - Set self.last_evaluation_date = max(triangle.development_periods)
            - Handle edge cases: zero denominators, single-year triangles
        """
        raise NotImplementedError(
            "fit stub — implement chain-ladder factor calculation"
        )

    def predict(self, triangle: LossTriangle) -> tuple[dict, pd.DataFrame]:
        """
        Apply chain-ladder factors to project ultimate losses.

        Starting from the most recent cumulative loss, multiplies by
        development factors for remaining development periods to estimate
        ultimate loss and remaining IBNR.

        Args:
            triangle: LossTriangle to project (typically partial triangle)

        Returns:
            Tuple of:
            - ultimate_losses: Dict[origin_year] -> projected ultimate loss
            - completed_triangle: DataFrame with full projected triangle

        TODO:
            - For each origin year in triangle:
              - Get latest cumulative loss in data
              - Multiply by remaining development factors
              - Project cumulative_loss to ultimate (latest dev period)
              - Store ultimate loss estimate
            - Construct completed_triangle DataFrame with projected values
            - Include both actual and projected portions
            - Return (ultimate_losses dict, completed_triangle DataFrame)
        """
        raise NotImplementedError(
            "predict stub — implement chain-ladder projection logic"
        )

    def get_ibnr(self, triangle: LossTriangle, ultimate_losses: dict) -> dict:
        """
        Calculate IBNR (Incurred But Not Reported) reserves.

        IBNR = Ultimate Loss - Last Evaluated Cumulative Loss

        Args:
            triangle: Original triangle (for latest cumulative values)
            ultimate_losses: Dict from predict() with ultimate estimates

        Returns:
            Dict[origin_year] -> IBNR reserve amount

        TODO:
            - For each origin year:
              - Get last cumulative loss from triangle
              - Calculate IBNR = ultimate_losses[year] - last_cumulative
              - Handle cases where origin_year not in latest eval (full development)
            - Return dict of IBNR by origin year
        """
        raise NotImplementedError("get_ibnr stub — implement IBNR calculation")

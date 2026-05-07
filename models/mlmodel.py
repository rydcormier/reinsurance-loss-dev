"""
Machine learning model for loss reserve prediction.

Flexible model architecture for predicting incremental losses (or ultimate losses)
from engineered features. Architecture is TBD (see fit() docstring for options).
Includes uncertainty quantification via prediction intervals.
"""

from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


class LossDevModel:
    """
    ML-based loss development model with uncertainty quantification.

    Predicts incremental or cumulative losses from engineered features.
    Supports multiple backends: gradient boosted trees (XGBoost/LightGBM),
    PyTorch MLP with dropout, or Gaussian process regression.

    Attributes:
        model: Fitted model object (architecture TBD)
        scaler: Feature normalization scaler (sklearn-style)
        uncertainty_method: Method for computing prediction intervals
    """

    def __init__(self, model=None, scaler=None):
        """
        Initialize LossDevModel.

        Args:
            model: Pre-trained model object or None
            scaler: Feature scaler or None
        """
        self.model = model
        self.scaler = scaler
        self.uncertainty_method: str = "None"  # Will be set during fit

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_split: float = 0.2,
        random_state: int = 42,
    ) -> "LossDevModel":
        """
        Fit the ML model to training data.

        Architecture options (choose one):

        Option 1: Gradient Boosted Trees (XGBoost or LightGBM)
            - Excellent for tabular data with mixed feature types
            - Native handling of missing values, categorical features
            - Feature importance and partial dependence plots
            - Fast training and inference
            - Uncertainty via quantile regression (predict 10th/90th percentiles)

        Option 2: PyTorch MLP with Dropout
            - Flexible architecture, good for learning complex interactions
            - Dropout-based epistemic uncertainty quantification
            - Requires feature normalization
            - Slower inference than trees but more extensible

        Option 3: Gaussian Process Regression
            - Built-in uncertainty via posterior variance
            - Expensive to scale to large datasets
            - Good for understanding confidence in predictions

        Args:
            X: Feature DataFrame [n_samples, n_features]
            y: Target variable (incremental or cumulative loss) [n_samples]
            val_split: Fraction of data for validation (default 0.2)
            random_state: Random seed for reproducibility

        Returns:
            self (for method chaining)

        TODO:
            - Choose architecture from the three options above (document choice)
            - Split X, y into train/val using val_split
            - Normalize features using features.engineering.normalize_features(fit=True)
            - Initialize model with architecture-specific hyperparameters
            - Train on X_train, y_train
            - Validate on X_val, y_val during training
            - Store trained model in self.model
            - Store scaler in self.scaler
            - Set self.uncertainty_method based on chosen architecture
            - Log training metrics (MAE, RMSE on val set)
        """
        raise NotImplementedError(
            "fit stub — implement model training (choose architecture in code)"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point predictions for loss amounts.

        Args:
            X: Feature DataFrame [n_samples, n_features]

        Returns:
            Predicted losses [n_samples]

        TODO:
            - Normalize X using stored self.scaler
            - Call self.model.predict(X_normalized)
            - Return predictions as numpy array
        """
        raise NotImplementedError("predict stub — implement point prediction")

    def predict_with_uncertainty(
        self, X: pd.DataFrame, lower: float = 0.1, upper: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.

        Returns point estimate plus lower and upper percentile bounds.
        Uncertainty method depends on architecture chosen in fit().

        Args:
            X: Feature DataFrame [n_samples, n_features]
            lower: Lower percentile bound (default 0.1 = 10th percentile)
            upper: Upper percentile bound (default 0.9 = 90th percentile)

        Returns:
            Tuple of (point_pred, lower_bound, upper_bound), each [n_samples]

        TODO (XGBoost/LightGBM implementation):
            - Train separate quantile regression models for lower/upper percentiles
            - Return (median prediction, 10th pct prediction, 90th pct prediction)

        TODO (PyTorch implementation):
            - Enable dropout during inference (Monte Carlo dropout)
            - Generate multiple forward passes (e.g., 100 samples)
            - Compute empirical percentiles from samples
            - Return (mean, percentile(lower), percentile(upper))

        TODO (GP implementation):
            - Call predict with return_std=True
            - Assume Gaussian posterior: lower/upper = mean ± std * z_score
            - Return bounds at requested percentiles
        """
        raise NotImplementedError(
            "predict_with_uncertainty stub — implement interval prediction"
        )

    def save(self, filepath: Path | str) -> None:
        """
        Serialize model and scaler to disk.

        Args:
            filepath: Path to save model (typically .pkl or .pt depending on backend)

        TODO:
            - Serialize self.model using pickle (XGBoost/LightGBM) or torch.save (PyTorch)
            - Also save self.scaler
            - Save metadata: architecture type, uncertainty_method, feature names
            - Use joblib for sklearn objects or pickle for custom code
        """
        raise NotImplementedError("save stub — implement model serialization")

    @classmethod
    def load(cls, filepath: Path | str) -> "LossDevModel":
        """
        Deserialize model and scaler from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded LossDevModel instance

        TODO:
            - Deserialize model from filepath
            - Deserialize scaler
            - Load metadata
            - Return cls(model=..., scaler=...)
        """
        raise NotImplementedError("load stub — implement model deserialization")

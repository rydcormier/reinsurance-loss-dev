"""
Evaluation script for loss development model.

Loads trained model and test data, generates predictions,
computes metrics, and compares to chain-ladder baseline.
Usage: python scripts/evaluate.py --model models/saved/model.pkl --data data/processed/test.parquet
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# TODO: Import actual modules once implemented
# from models.mlmodel import LossDevModel
# from models.chainladder import ChainLadder
# from evaluation.metrics import evaluate, compare_to_chainladder, divergence_flag

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Evaluate ML model on test data."""
    parser = argparse.ArgumentParser(
        description="Evaluate loss development model"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model pickle file",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to test features parquet file",
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=0.15,
        help="Divergence flag threshold (default 0.15 = 15%)",
    )

    args = parser.parse_args()

    logger.info(f"Loading model from {args.model}")
    logger.info(f"Loading test data from {args.data}")

    # TODO:
    #   - Load model: ml_model = LossDevModel.load(args.model)
    #   - Load test data: test_df = pd.read_parquet(args.data)
    #   - Separate X_test, y_test
    #   - Generate predictions: y_pred, y_lower, y_upper = ml_model.predict_with_uncertainty(X_test)
    #   - Evaluate ML model: ml_metrics = evaluate(y_test, y_pred, y_lower, y_upper)
    #   - Initialize and fit chain-ladder baseline on training data (or use pre-fitted)
    #   - Get CL predictions: cl_pred = chainladder_model.predict(...)
    #   - Evaluate CL: cl_metrics = evaluate(y_test, cl_pred)
    #   - Compare: comparison = compare_to_chainladder(ml_metrics, cl_metrics)
    #   - For each sample, compute divergence_flag()
    #   - Print formatted results table
    #   - Print verdict: "PRODUCTION READY" | "INVESTIGATE" | "AVOID"
    #   - Log to file (optional)

    logger.warning("evaluate.py is a stub — implement actual evaluation logic")
    raise NotImplementedError("evaluate.py stub — implement evaluation pipeline")


if __name__ == "__main__":
    main()

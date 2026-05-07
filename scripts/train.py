"""
Training script for loss development model.

Loads processed features, trains the ML model, and saves to disk.
Usage: python scripts/train.py --data data/processed/features.parquet --output models/saved/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# TODO: Import actual modules once implemented
# from features.engineering import normalize_features
# from models.mlmodel import LossDevModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train ML model on processed features."""
    parser = argparse.ArgumentParser(
        description="Train loss development model"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to processed features parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/saved"),
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Train/val split ratio",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading features from {args.data}")

    # TODO:
    #   - Load features from args.data using pd.read_parquet()
    #   - Separate X (features) and y (target variable)
    #   - Initialize LossDevModel()
    #   - Call model.fit(X, y, val_split=args.val_split, random_state=args.random_seed)
    #   - Save model: model.save(output_dir / "model.pkl")
    #   - Log training metrics and model summary
    #   - Log saved path

    logger.warning("train.py is a stub — implement actual training logic")
    raise NotImplementedError("train.py stub — implement training pipeline")


if __name__ == "__main__":
    main()

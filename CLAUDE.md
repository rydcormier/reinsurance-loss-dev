# CLAUDE.md – Project Context and Scaffold Pattern

## Project Overview

**reinsurance-loss-dev** is a modular Python framework for loss reserve prediction and reserve adequacy assessment in reinsurance. It combines traditional actuarial methods (chain-ladder) with machine learning to predict ultimate losses and remaining IBNR (Incurred But Not Reported) reserves from historical loss development triangles.

### Key Features

- **Loss Triangle Ingestion**: Parse and validate loss triangles from CSV/Excel
- **Feature Engineering**: Transform raw triangles into modeling-ready panels
- **Chain-Ladder Baseline**: Traditional actuarial method for benchmarking
- **ML Model Framework**: Flexible architecture (XGBoost, PyTorch, Gaussian Process)
- **Uncertainty Quantification**: Prediction intervals (10th/90th percentiles)
- **Divergence Detection**: Flag when ML and traditional estimates differ significantly
- **REST API**: FastAPI for production serving
- **Comprehensive Evaluation**: Metrics, comparisons, and governance support

---

## Scaffold Pattern

This project uses a **stub-first, interface-driven architecture**:

1. **Every function has a complete docstring** explaining:
   - What it does
   - Input parameters and types
   - Expected output and return types
   - Any edge cases or assumptions

2. **Every stub includes a clear TODO comment** describing implementation requirements:
   - What calculations or logic should be performed
   - Any dependencies on other modules
   - Specific validation or error handling needed

3. **Real interfaces, not magic**:
   - Actual dataclasses and Pydantic models that will work when stubs are filled in
   - Type hints throughout for IDE support
   - No placeholder lambdas or dummy return values

4. **Structure mirrors insurance-doc-agent patterns**:
   - Modular directory layout (ingestion, features, models, evaluation, api, scripts)
   - Pydantic schemas for API requests/responses
   - FastAPI lifespan context manager for app startup
   - Comprehensive docstrings and TODO comments

### Benefits of This Pattern

- **Clarity**: Everyone knows what needs to be implemented and why
- **Type Safety**: IDE support catches errors early
- **Testability**: Tests can validate interfaces before implementations exist
- **Modularity**: Clear separation of concerns makes maintenance easier
- **Documentation**: Docstrings and TODOs serve as specifications

---

## Module Guide

### `ingestion/triangles.py`

**Role**: Parse and validate loss triangles from various formats.

**Key Types**:
- `LossTriangle`: Dataclass holding origin_years, development_periods, incremental_losses, cumulative_losses, optional premium

**Key Functions**:
- `parse_csv()` / `parse_excel()`: Read triangles from files
- `validate()`: Check for data quality issues
- `to_dataframe()`: Convert to long format for modeling

**Implementation Notes**:
- First column of CSV = origin_year (accident/exposure year)
- Subsequent columns = development periods (12, 24, 36, ... months)
- Cell values = cumulative losses
- Must handle missing values (incomplete recent cohorts)
- Incremental losses = cumulative[t] - cumulative[t-1]

---

### `features/engineering.py`

**Role**: Transform loss triangles into feature panels for machine learning.

**Key Function**:
- `build_feature_panel()`: Create one row per (origin_year, dev_period) cell with engineered features

**Feature Types**:
- **Raw**: incremental_loss, cumulative_loss
- **Lags**: prior period losses, growth rates
- **Transforms**: log(loss), development_age = dev_period / 12
- **Fixed Effects**: one-hot dev_period, one-hot origin_year
- **Exposures**: loss-to-premium ratio (if premium available)

**Output Shape**: [n_origin_years × n_dev_periods, n_features]

---

### `models/chainladder.py`

**Role**: Traditional actuarial reserve estimation method (baseline).

**Key Class**: `ChainLadder`

**Methods**:
- `fit(triangle)`: Calculate development factors
- `predict(triangle)`: Project ultimate losses and completed triangle
- `get_ibnr(triangle, ultimate_losses)`: Calculate reserve amounts

**Algorithm**:
- Development factor[t] = Sum(cumulative_loss[:, t]) / Sum(cumulative_loss[:, t-1])
- Apply factors to latest losses to project ultimate

**Role in System**: Benchmark for ML model. Must be beaten on MAE and RMSE for production readiness.

---

### `models/mlmodel.py`

**Role**: Machine learning model for loss prediction with uncertainty.

**Key Class**: `LossDevModel`

**Methods**:
- `fit(X, y)`: Train on features and target
- `predict(X)`: Point estimates
- `predict_with_uncertainty(X)`: Point + 10th/90th percentile bounds
- `save()` / `load()`: Serialization

**Architecture Options** (choose one during implementation):
1. **Gradient Boosted Trees (XGBoost/LightGBM)**
   - Fast, handles mixed features
   - Uncertainty via quantile regression
   - Feature importance available

2. **PyTorch MLP with Dropout**
   - Flexible, learns complex interactions
   - Uncertainty via Monte Carlo dropout
   - Requires normalization

3. **Gaussian Process Regression**
   - Built-in uncertainty via posterior variance
   - Computationally expensive for large data
   - Good confidence calibration

---

### `evaluation/metrics.py`

**Role**: Compute metrics, compare models, and detect divergences.

**Key Types**:
- `ReserveMetrics`: mae, rmse, mape, bias, coverage_80, coverage_90

**Key Functions**:
- `evaluate(y_true, y_pred, y_lower, y_upper)`: Compute metrics
- `compare_to_chainladder(ml_metrics, cl_metrics)`: Structured comparison
- `divergence_flag(ml_estimate, cl_estimate, threshold=0.15)`: Flag significant deviations

**Production-Readiness**:
- ML model must beat chain-ladder on **both MAE and RMSE**
- Divergence flag (15% default) triggers actuarial review

---

### `api/app.py`

**Role**: REST API serving predictions and managing triangles.

**Framework**: FastAPI with Pydantic schemas

**Key Endpoints**:
- `GET /health`: Health check
- `POST /triangles/upload`: Upload triangle, validate, return triangle_id
- `POST /predict`: Generate predictions with confidence intervals and divergence flags
- `GET /triangles`: List uploaded triangles
- `GET /triangles/{triangle_id}`: Retrieve triangle and last prediction

**Design Pattern**: Lifespan context manager for startup/shutdown, structured JSON responses, HTTPException for errors.

---

### `scripts/train.py` and `scripts/evaluate.py`

**Role**: CLI interfaces for training and evaluation workflows.

**train.py**:
- Loads processed features from parquet
- Calls `LossDevModel.fit()`
- Saves model to disk

**evaluate.py**:
- Loads trained model
- Loads test data
- Runs ML and chain-ladder predictions
- Prints evaluation metrics and comparison
- Flags divergences

---

### `tests/test_triangles.py`

**Role**: Unit tests for ingestion and feature engineering.

**Pattern**:
- Synthetic 4x4 loss triangle fixture (hardcoded for deterministic testing)
- Tests for `parse_csv()`, `validate()`, `to_dataframe()`, `build_feature_panel()`
- Test interface correctness, not implementation details

---

## Stub Completion Workflow

Suggested order for filling in TODOs:

1. **Ingestion** (`ingestion/triangles.py`): Parse triangles, validate data
2. **Features** (`features/engineering.py`): Engineer features from triangles
3. **Baseline** (`models/chainladder.py`): Implement chain-ladder
4. **ML Model** (`models/mlmodel.py`): Choose architecture, implement fit/predict
5. **Evaluation** (`evaluation/metrics.py`): Compute metrics and comparisons
6. **Tests** (`tests/test_triangles.py`): Update to verify implementations
7. **API** (`api/app.py`): Wire up endpoints using above modules
8. **Scripts** (`scripts/train.py`, `scripts/evaluate.py`): Test end-to-end workflows

---

## Configuration

Environment variables (see `.env.example`):
- `MODEL_PATH`: Path to saved ML model
- `DATA_DIR`: Root data directory
- `API_PORT`: FastAPI server port
- `DIVERGENCE_THRESHOLD`: Threshold for flagging divergences (default 0.15)

---

## Running the Project

```bash
# Install
make install

# Test
make test

# Train (after implementing stubs and creating training data)
make train

# Evaluate
make evaluate

# Serve
make serve
```

---

## Notes

- This is a **real, production-capable architecture** — not a toy.
- Every interface is designed to work when stubs are completed.
- Tests are written to validate interfaces, not implementations.
- The chain-ladder baseline is essential: it's not just a reference, it's a governance requirement.
- Divergence flags support actuarial workflows: when the ML model disagrees significantly with traditional estimates, humans review.

---

## References

- Similar pattern: [insurance-doc-agent](https://github.com/rydcormier/insurance-doc-agent)
- Actuarial background: See `docs/actuarial_background.md`
- Architecture details: See `ARCHITECTURE.md`

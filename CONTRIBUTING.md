# Contributing

This document outlines the development workflow, stub completion strategy, and definition of production-readiness for the **reinsurance-loss-dev** project.

---

## Overview

This project is structured as a **scaffold with stub implementations**. Every module has:
- Complete docstrings and type hints
- Clear TODO comments describing required implementation
- Real, working interfaces that will function once stubs are filled in

Your role is to complete these TODOs in a logical order, ensuring each module works correctly before moving to the next.

---

## Development Workflow

### 1. Setup

```bash
# Clone repository
git clone <repo-url>
cd reinsurance-loss-dev

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# Install dependencies
make install
```

### 2. Choose Implementation

Before starting, decide on the ML model architecture in `models/mlmodel.py`:

**Option A: Gradient Boosted Trees (XGBoost/LightGBM)** ← Recommended for tabular data
- Fast, handles mixed feature types
- Native missing value handling
- Uncertainty via quantile regression

**Option B: PyTorch MLP with Dropout**
- Flexible, learns complex interactions
- Uncertainty via Monte Carlo dropout
- Requires feature normalization

**Option C: Gaussian Process Regression**
- Built-in uncertainty
- Computationally expensive at scale
- Excellent confidence calibration

Document your choice in `models/mlmodel.py` docstring before implementing.

### 3. Stub Completion Order

Complete stubs in this order to ensure dependencies are satisfied:

#### Phase 1: Data Pipeline (Foundations)

1. **`ingestion/triangles.py`**
   - Implement `parse_csv()`: Read CSV with origin_year rows, dev_period columns
   - Implement `parse_excel()`: Read Excel sheets
   - Implement `validate()`: Check for NaN, negatives, non-monotonic cumulative losses
   - Implement `to_dataframe()`: Convert to long format [n_cells, 4 columns]
   - **Test**: `pytest tests/test_triangles.py::test_parse_csv_*`
   - **Check**: Verify synthetic triangle loads and passes validation

2. **`features/engineering.py`**
   - Implement `build_feature_panel()`: Create features from triangle
   - Add lag features, log transforms, fixed effects, loss-to-premium ratio
   - Handle edge cases (first dev period has NaN lags)
   - **Test**: `pytest tests/test_triangles.py::test_build_feature_panel_*`
   - **Check**: Verify feature panel has correct shape and values

#### Phase 2: Modeling (Core)

3. **`models/chainladder.py`**
   - Implement `fit()`: Calculate volume-weighted development factors
   - Implement `predict()`: Apply factors to project ultimate losses
   - Implement `get_ibnr()`: Calculate reserve amounts
   - **Test**: Create small test triangle, verify factors are > 1.0 for early periods
   - **Check**: Compare predicted ultimates to inputs (should be >= cumulative)

4. **`models/mlmodel.py`** (Choose Architecture)
   - Implement `fit()`: Train model on features and targets
   - Implement `predict()`: Generate point predictions
   - Implement `predict_with_uncertainty()`: Point + percentile bounds
   - Implement `save()` / `load()`: Serialization
   - **Test**: Verify predictions have reasonable magnitude
   - **Check**: Compare to chain-ladder baseline (may not beat it initially)

#### Phase 3: Evaluation (Metrics)

5. **`evaluation/metrics.py`**
   - Implement `evaluate()`: Compute mae, rmse, mape, bias, coverage
   - Implement `compare_to_chainladder()`: Structured comparison
   - Implement `divergence_flag()`: Flag 15%+ deviations
   - **Test**: Create known predictions and verify metrics
   - **Check**: ML metrics vs CL metrics on test data

#### Phase 4: Integration (API & Scripts)

6. **`scripts/train.py`**
   - Implement data loading from parquet
   - Implement model training loop
   - Implement model saving to disk
   - **Test**: `python scripts/train.py --data data/processed/features.parquet`
   - **Check**: Model file created and loadable

7. **`scripts/evaluate.py`**
   - Implement model loading
   - Implement prediction generation
   - Implement metrics printing and comparison
   - Implement divergence flag output
   - **Test**: `python scripts/evaluate.py --model models/saved/model.pkl`
   - **Check**: Formatted output with verdict (production_ready / investigate / avoid)

8. **`api/app.py`**
   - Implement `/health` endpoint
   - Implement `/triangles/upload` endpoint (parse, validate, store)
   - Implement `/predict` endpoint (features → predictions → comparison)
   - Implement `/triangles` and `/triangles/{triangle_id}` endpoints
   - **Test**: Use curl or Postman to test endpoints
   - **Check**: API returns correct JSON schema

9. **`tests/test_triangles.py`**
   - Update test fixtures to use completed functions
   - Verify tests pass with real implementations
   - Add additional tests as needed

---

## Definition of Production-Readiness

A model is **production-ready** when:

### Mandatory Criteria

1. **ML beats Chain-Ladder on MAE**: `ml_mae < cl_mae`
2. **ML beats Chain-Ladder on RMSE**: `ml_rmse < cl_rmse`
3. **All tests pass**: `pytest tests/ -v`
4. **API endpoints working**: Health, upload, predict, list, retrieve
5. **No unhandled exceptions**: Graceful error handling with HTTPException

### Strongly Recommended

6. **Coverage metrics acceptable**: 80% coverage close to 0.80, 90% coverage close to 0.90
7. **Bias near zero**: Predictions not systematically over/under-estimating
8. **Divergence flags reasonable**: < 20% of predictions flagged for review
9. **Documentation complete**: README, ARCHITECTURE, actuarial_background all filled in
10. **Model serialization working**: Can save and load without errors

### Governance

- **Divergence flags** (15% threshold) trigger manual actuarial review
- **Production predictions** go to `/api/predict` endpoint
- **Batch predictions** use `scripts/evaluate.py` for analysis
- **Model updates** require retraining and re-evaluation with new test set

---

## Running Development Commands

```bash
# Run tests (validate implementations)
make test
# or: pytest tests/ -v

# Train model (create data/processed/features.parquet first)
make train
# or: python scripts/train.py --data data/processed/features.parquet --output models/saved/

# Evaluate on test set
make evaluate
# or: python scripts/evaluate.py --model models/saved/model.pkl --data data/processed/test.parquet

# Start API server
make serve
# or: uvicorn api.app:app --reload
# Then: curl http://localhost:8000/health

# Code quality
make lint
# or: black ., isort ., pylint ingestion/ ...

# Clean build artifacts
make clean
```

---

## Architecture Decision Notes

### Why Chain-Ladder Baseline?

Chain-ladder is the industry standard for reserve estimation. It's:
- **Well-understood** by actuaries
- **Transparent**: Easy to explain development factors
- **Conservative**: Tends to be adequate for typical triangles
- **Benchmark**: ML must exceed it to justify additional complexity

### Why Divergence Flag at 15%?

- **Too tight (< 5%)**: Flags many benign differences, reduces actionability
- **Too loose (> 25%)**: Misses significant deviations
- **15%**: Historical experience shows this is a reasonable governance threshold

### Why Prediction Intervals?

- **Point estimates alone are insufficient** for reserve setting
- **Intervals support risk quantification** (e.g., 80th percentile reserve)
- **Coverage metrics** (actual % falling in interval) validate calibration

---

## Code Style

- **Type hints**: Use `type | str` (Python 3.10+) or `Union[type, str]` (older)
- **Docstrings**: Google-style with Args, Returns, Raises
- **Format**: Use `black` and `isort` (see Makefile lint target)
- **Comments**: Explain *why*, not *what* (code should be self-documenting)

---

## Testing Strategy

1. **Unit tests** (`tests/test_triangles.py`): Test individual functions
2. **Integration tests** (`scripts/evaluate.py`): End-to-end workflow
3. **API tests** (manual or pytest-fixtures): Endpoint validation
4. **Synthetic data**: Use hardcoded triangle fixture for reproducibility

Test expectations:
- Tests validate **interfaces** (correct input/output types and shapes)
- Tests validate **business logic** (metrics computed correctly)
- Tests do **not** validate model *performance* (that's evaluation.py's job)

---

## Deployment Checklist

Before deploying to production:

- [ ] ML beats CL on MAE and RMSE
- [ ] All tests pass
- [ ] API endpoints tested with curl/Postman
- [ ] Model serialization tested (save/load)
- [ ] Divergence flags reviewed and threshold justified
- [ ] Documentation updated
- [ ] `.env` configured with production paths
- [ ] Logging configured to INFO level
- [ ] Error handling comprehensive

---

## Questions & Support

Refer to:
- **Architecture questions**: See `ARCHITECTURE.md`
- **Actuarial questions**: See `docs/actuarial_background.md`
- **API questions**: See `README.md` API Reference
- **Module questions**: See `CLAUDE.md` Module Guide

---

## License

Contributions to this project are assumed to be under the same license as the project. See `LICENSE` file.

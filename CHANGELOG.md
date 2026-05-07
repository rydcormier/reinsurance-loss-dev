# Changelog

All notable changes to this project will be documented in this file.

---

## [0.1.0] – 2026-05-07

### Initial Project Scaffold

**Added**:
- Complete modular directory structure: `ingestion/`, `features/`, `models/`, `evaluation/`, `api/`, `scripts/`, `tests/`, `notebooks/`, `docs/`
- **Ingestion module** (`ingestion/triangles.py`):
  - `LossTriangle` dataclass with origin_years, development_periods, losses, premium
  - `parse_csv()` and `parse_excel()` stubs for reading triangles from files
  - `validate()` stub for data quality checks (missing values, negatives, monotonicity)
  - `to_dataframe()` stub for long-format conversion
- **Feature engineering** (`features/engineering.py`):
  - `build_feature_panel()` stub for row-per-cell feature transformation
  - Lag features, log transforms, fixed effects, loss-to-premium ratio
- **Models** (`models/chainladder.py` and `models/mlmodel.py`):
  - `ChainLadder` class with `fit()` and `predict()` for baseline estimation
  - `LossDevModel` class with `fit()`, `predict()`, `predict_with_uncertainty()` for ML
  - Architecture options documented: XGBoost, PyTorch MLP, Gaussian Process
- **Evaluation** (`evaluation/metrics.py`):
  - `ReserveMetrics` dataclass (mae, rmse, mape, bias, coverage_80, coverage_90)
  - `evaluate()` stub for metric computation
  - `compare_to_chainladder()` stub for structured comparison
  - `divergence_flag()` stub for detecting significant deviations (15% threshold)
- **FastAPI serving** (`api/app.py`):
  - REST endpoints: `/health`, `/triangles/upload`, `/predict`, `/triangles`, `/triangles/{triangle_id}`
  - Pydantic request/response schemas
  - Lifespan context manager for startup/shutdown
  - Structured JSON responses and HTTPException handling
- **CLI scripts** (`scripts/train.py`, `scripts/evaluate.py`):
  - Training script with argparse interface
  - Evaluation script with metrics and comparison output
- **Testing** (`tests/test_triangles.py`):
  - Unit test stubs with synthetic 4x4 loss triangle fixture
  - Tests for parse_csv, validate, to_dataframe, build_feature_panel
- **Configuration**:
  - `requirements.txt` with pandas, numpy, scipy, scikit-learn, xgboost, torch, fastapi, uvicorn, pydantic, pytest, chainladder
  - `.env.example` with MODEL_PATH, DATA_DIR, LOG_LEVEL, DIVERGENCE_THRESHOLD
  - `Makefile` with targets: install, test, train, evaluate, serve, lint, clean
- **Documentation**:
  - `README.md` with project overview, quickstart, API reference, metrics explanation
  - `CLAUDE.md` with project context, scaffold pattern, module guide, completion workflow
  - `CHANGELOG.md` (this file)
  - `CONTRIBUTING.md` with development workflow and production-ready criteria
  - `ARCHITECTURE.md` with data flow, design decisions, divergence flag logic
  - `docs/actuarial_background.md` with loss development problem, chain-ladder explanation, terminology glossary
- **Sample data**: `data/raw/sample_triangle.csv` with 5x5 synthetic Workers' Compensation triangle

**Notes**:
- All functions are stubs with comprehensive docstrings and clear TODO comments
- Interfaces are real and production-capable; implementations to be completed
- Chain-ladder baseline must be beaten on MAE and RMSE for production deployment
- Divergence flag supports actuarial review workflows
- Project structure follows [insurance-doc-agent](https://github.com/rydcormier/insurance-doc-agent) patterns

---

## Future Versions

### [0.2.0] (planned)
- Implement ingestion module (parse_csv, parse_excel, validate, to_dataframe)
- Implement feature engineering pipeline
- Implement chain-ladder baseline

### [0.3.0] (planned)
- Implement ML model framework (choose architecture)
- Implement model training and evaluation

### [0.4.0] (planned)
- Integrate with FastAPI endpoints
- Deploy evaluation on test data

### [1.0.0] (planned)
- Full production deployment
- ML model beats chain-ladder on MAE and RMSE
- Comprehensive documentation and examples

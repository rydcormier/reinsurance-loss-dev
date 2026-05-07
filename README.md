# Reinsurance Loss Development & Reserve Adequacy Prediction

A modular Python framework for predicting loss reserves and assessing reserve adequacy using machine learning and traditional actuarial methods. Combines chain-ladder baseline with ML-based predictions, includes uncertainty quantification, and provides FastAPI REST endpoints for production deployment.

## Project Overview

This system addresses the loss development prediction problem: given historical loss triangles, estimate ultimate losses and remaining IBNR (Incurred But Not Reported) reserves. The framework:

- **Parses and validates** loss triangles from CSV/Excel formats
- **Engineers features** from raw loss data for machine learning
- **Provides a chain-ladder baseline** for comparison
- **Trains ML models** (architecture TBD: XGBoost, PyTorch, or Gaussian Process)
- **Generates predictions with confidence intervals** (10th and 90th percentiles)
- **Flags divergences** between ML and traditional estimates for actuarial review
- **Serves predictions via REST API** with FastAPI

**Production-Ready Criteria:** The ML model must beat chain-ladder on both **MAE** and **RMSE** before production deployment. Divergence flags and prediction intervals support manual actuarial review workflows.

---

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `data/raw/` | Source loss triangle data (CSV/Excel) — gitignored |
| `data/processed/` | Engineered features and model outputs — gitignored |
| `ingestion/` | Triangle parsing and validation (CSV, Excel) |
| `features/` | Feature engineering: triangle → modeling-ready panel |
| `models/` | Model implementations: chain-ladder baseline + ML |
| `evaluation/` | Metrics, comparisons, and divergence detection |
| `api/` | FastAPI serving layer with REST endpoints |
| `scripts/` | CLI for training and evaluation |
| `tests/` | Unit tests with synthetic triangle fixture |
| `notebooks/` | EDA and prototyping (Jupyter) |
| `docs/` | Architecture, background, and design docs |

---

## Quick Start

### 1. Install Dependencies

```bash
make install
# or: pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your paths/settings
```

### 3. Prepare Data

Place loss triangles in CSV or Excel format in `data/raw/`. Sample format:

```csv
origin_year,12,24,36,48,60
2019,1000000,1580000,1720000,1785000,1820000
2020,950000,1501000,1640000,1708000,
2021,920000,1460000,1600000,,
...
```

### 4. Train Model

```bash
python scripts/train.py \
  --data data/processed/features.parquet \
  --output models/saved/ \
  --val-split 0.2 \
  --random-seed 42
```

Or use the Makefile:
```bash
make train
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
  --model models/saved/model.pkl \
  --data data/processed/test.parquet \
  --divergence-threshold 0.15
```

Or:
```bash
make evaluate
```

### 6. Run Tests

```bash
make test
# or: pytest tests/ -v
```

### 7. Serve API

```bash
make serve
# or: uvicorn api.app:app --reload
```

The API will be available at `http://localhost:8000`.

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — returns `{"status": "healthy"}` |
| POST | `/triangles/upload` | Upload triangle (CSV/Excel), returns `triangle_id` and validation results |
| POST | `/predict` | Generate predictions for a triangle; returns ML predictions, CL baseline, divergence flags |
| GET | `/triangles` | List all uploaded triangles with metadata |
| GET | `/triangles/{triangle_id}` | Retrieve specific triangle metadata and last prediction |

### Example: Upload Triangle

```bash
curl -X POST http://localhost:8000/triangles/upload \
  -F "file=@data/raw/sample_triangle.csv" \
  -F "line_of_business=Workers Compensation"
```

Response:
```json
{
  "triangle_id": "abc-def-123",
  "line_of_business": "Workers Compensation",
  "validation": {
    "is_valid": true,
    "errors": []
  },
  "origin_years": [2019, 2020, 2021, 2022, 2023],
  "development_periods": [12, 24, 36, 48, 60]
}
```

### Example: Generate Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"triangle_id": "abc-def-123", "use_uncertainty": true}'
```

Response:
```json
{
  "triangle_id": "abc-def-123",
  "point_estimate": {
    "2019": 1820000,
    "2020": 1750000,
    ...
  },
  "lower_bound_10pct": {...},
  "upper_bound_90pct": {...},
  "chain_ladder_estimate": {...},
  "divergence_flags": {
    "2019": false,
    "2020": true,
    ...
  },
  "ml_vs_cl_comparison": {
    "mae_improvement_pct": 12.5,
    "rmse_improvement_pct": 15.3,
    ...
  }
}
```

---

## Evaluation Metrics

The framework tracks the following metrics:

- **MAE** (Mean Absolute Error): Average absolute prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Relative error as %
- **Bias**: Mean signed error (positive = underprediction)
- **Coverage 80% / 90%**: Fraction of actuals within 80% / 90% prediction intervals

### Chain-Ladder Baseline

The chain-ladder method is the industry standard for reserve estimation. It:
- Calculates volume-weighted age-to-age development factors from historical data
- Applies these factors to project ultimate losses
- Provides a benchmark the ML model must exceed

### Divergence Flag

When ML and chain-ladder estimates diverge by > 15% (configurable), the result is flagged for manual actuarial review. This supports governance workflows by highlighting cases where the ML model significantly deviates from traditional estimates.

---

## Development Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Stub completion order and priorities
- Definition of "production-ready"
- How to run development commands

See [ARCHITECTURE.md](ARCHITECTURE.md) for:
- Data flow diagrams
- Design decision rationale
- Feature engineering details

See [docs/actuarial_background.md](docs/actuarial_background.md) for:
- Loss development problem explanation
- Chain-ladder method and limitations
- Glossary of actuarial terms

---

## Project Status

**Current Version:** 0.1.0 (Scaffold)

This is a stub implementation with real interfaces and comprehensive TODO comments. The structure is production-ready; the implementations are to be completed. See [CHANGELOG.md](CHANGELOG.md) for history.

---

## Dependencies

See [requirements.txt](requirements.txt) for complete list:
- **Data**: pandas, numpy, scipy, pyarrow
- **ML**: scikit-learn, xgboost, torch
- **API**: fastapi, uvicorn, pydantic
- **Actuarial**: chainladder
- **Testing**: pytest
- **Utilities**: python-dotenv, openpyxl

---

## Configuration

Environment variables (see [.env.example](.env.example)):

```bash
MODEL_PATH=models/saved/model.pkl
DATA_DIR=data/
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
DIVERGENCE_THRESHOLD=0.15
```

---

## Author & License

Scaffold structure follows patterns from [insurance-doc-agent](https://github.com/rydcormier/insurance-doc-agent).

See [LICENSE](LICENSE) for licensing details.

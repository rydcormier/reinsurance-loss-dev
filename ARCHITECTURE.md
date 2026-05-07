# Architecture

This document describes the system architecture, data flow, design decisions, and technical rationale for **reinsurance-loss-dev**.

---

## System Overview

The system predicts loss reserves using a hybrid approach:
1. **Traditional**: Chain-ladder baseline for benchmarking
2. **Modern**: ML model for potentially better accuracy
3. **Governance**: Divergence detection for actuarial review

```
Loss Triangle (CSV/Excel)
         ↓
    Ingestion (validation)
         ↓
Feature Engineering (feature panel)
         ↓
     ┌────────────┬────────────┐
     ↓            ↓
 Chain-Ladder   ML Model
     ↓            ↓
 CL Prediction  ML Prediction
     ↓            ↓
  Evaluation (metrics)
     ↓
 Comparison & Divergence Flag
     ↓
  API Response / Review
```

---

## Data Flow

### Stage 1: Ingestion

**Input**: CSV or Excel file with loss triangle

**Format** (CSV):
```csv
origin_year,12,24,36,48,60
2019,1000000,1580000,1720000,1785000,1820000
2020,950000,1501000,1640000,1708000,
2021,920000,1460000,1600000,,
...
premium,5000000,4800000,4600000,4400000,4200000
```

**Processing** (`ingestion/triangles.py`):
- Parse origin_year from first column
- Extract development_periods from header
- Read cumulative losses into matrix
- Calculate incremental losses (cumulative[t] - cumulative[t-1])
- Extract premium row (optional)
- Validate: no NaN, no negatives, cumulative monotonic non-decreasing

**Output**: `LossTriangle` dataclass
```python
@dataclass
class LossTriangle:
    line_of_business: str
    origin_years: List[int]
    development_periods: List[int]
    incremental_losses: np.ndarray
    cumulative_losses: np.ndarray
    premium: np.ndarray = None
```

**Validation Rules**:
1. No NaN in complete rows (mature cohorts should have full development history)
2. No negative losses
3. Cumulative losses ≥ 0 and non-decreasing by development period
4. Origin years sorted ascending
5. Development periods sorted ascending

---

### Stage 2: Feature Engineering

**Input**: `LossTriangle` object

**Processing** (`features/engineering.py`):

Transform triangle to row-per-cell feature panel analogous to the policy-quarter panel in insurance modeling.

**Output Shape**: [n_cells, n_features]
- n_cells = number of non-NaN cells in cumulative_losses matrix
- Typical 5x5 triangle → 15 cells (upper triangular)

**Features by Category**:

| Category | Features | Purpose |
|----------|----------|---------|
| **Raw** | incremental_loss, cumulative_loss | Actual observed losses |
| **Lags** | incremental_loss_lag1, cumulative_loss_lag1 | Prior period losses for growth calc |
| **Growth** | incremental_loss_growth | % change from prior period |
| **Log** | log_incremental_loss, log_cumulative_loss | Scale reduction, handles range |
| **Time** | development_age = dev_period/12 | Years of development (continuous) |
| **Fixed Effects** | dev_period_fe_*, origin_year_fe_* | Categorical effects via one-hot |
| **Exposures** | loss_to_premium_ratio | Loss as % of underwritten premium |
| **Meta** | line_of_business | Categorical identifier |

**Edge Cases**:
- First development period: incremental_loss_lag1 = NaN
- Log transforms: use log(x + 1) to handle zeros
- Missing premium: loss_to_premium_ratio = NaN

**Feature Panel Output**:
```
origin_year dev_period cumulative incremental age log_cum log_inc lag1_cum growth ... dev_fe_24 year_fe_2019 ltp_ratio lob
2019        12         1000000    1000000      1.0 13.8    13.8    NaN      NaN      1           1           0.20      WC
2019        24         1580000    580000       2.0 14.3    13.3    13.8     -0.42    0           1           0.316     WC
...
```

---

### Stage 3: Modeling

#### Path A: Chain-Ladder Baseline (`models/chainladder.py`)

**Algorithm**:

1. **Fit Phase** (on historical triangle):
   ```
   For each development period t (except first):
       Factor[t] = Sum(cumulative_losses[:, t]) / Sum(cumulative_losses[:, t-1])
   ```

   Example: If column 24 sums to 10M and column 12 sums to 8M, then Factor[24] = 1.25

2. **Predict Phase** (on recent/incomplete triangle):
   ```
   For each origin year i and remaining dev periods t:
       projected_cumulative[i, t] = cumulative[i, t-1] * Factor[t]
   Ultimate[i] = projected_cumulative[i, last_period]
   IBNR[i] = Ultimate[i] - last_observed_cumulative[i]
   ```

**Rationale**:
- Industry standard; understood by all actuaries
- Transparent: factors are interpretable (e.g., "12-to-24 month factor is 1.4")
- Provides benchmark for ML model
- Conservative and empirical

**Implementation Notes**:
- Use volume-weighted average (not arithmetic mean) for factors
- Handle single-year triangles (no prior cohorts for comparison)
- Preserve factors for diagnostics

#### Path B: ML Model (`models/mlmodel.py`)

**Architecture Options** (choose one):

1. **Gradient Boosted Trees (XGBoost/LightGBM)** ← Recommended
   - Train on feature panel (X, y)
   - y = target variable (incremental loss or cumulative loss or ultimate loss, TBD)
   - Hyperparameters: depth, learning_rate, n_estimators (tune on val set)
   - Uncertainty: Train separate quantile models for 10th and 90th percentiles
   - Advantages: Fast, handles mixed features, feature importance, robust

2. **PyTorch MLP with Dropout**
   - Architecture: Input → Dense → ReLU → Dropout → Dense → ... → Output
   - Normalize features: mean=0, std=1 (sklearn StandardScaler)
   - Train with backprop, validate on holdout set
   - Uncertainty: Monte Carlo dropout (forward pass with dropout enabled 100× at test time)
   - Compute empirical percentiles from 100 samples
   - Advantages: Flexible, learns complex interactions, extensible

3. **Gaussian Process Regression**
   - Kernel: RBF or Matern (tune on validation set)
   - Training: Fit covariance matrix (expensive for large n)
   - Prediction: Posterior mean + variance
   - Uncertainty: Gaussian posterior; percentiles from CDF
   - Advantages: Built-in uncertainty, excellent calibration, but slow for n > 10k

**Target Variable Choice** (TBD during implementation):
- Option A: `y = incremental_loss` (predict next period's development)
- Option B: `y = cumulative_loss` (predict cumulative at dev period)
- Option C: `y = ultimate_loss` (predict final ultimate directly)

Recommend Option A (incremental) because:
- Naturally matches data structure (row per cell)
- Easier to validate incrementals sum correctly
- Can aggregate to cumulative via cumsum

**Hyperparameter Tuning**:
- Use val_split=0.2 from training data
- Metric: RMSE on validation set
- Early stopping if validation metric plateaus

---

### Stage 4: Evaluation

**Metrics Computed** (`evaluation/metrics.py`):

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | `mean(\|y_true - y_pred\|)` | Average error in dollars |
| **RMSE** | `sqrt(mean((y_true - y_pred)^2))` | Penalizes large errors |
| **MAPE** | `mean(\|y_true - y_pred\| / \|y_true\|) × 100` | Relative error % |
| **Bias** | `mean(y_true - y_pred)` | Systematic over/under-prediction |
| **Coverage 80%** | `fraction(y_lower ≤ y_true ≤ y_upper)` | Should be ≈ 0.80 |
| **Coverage 90%** | `fraction(y_lower ≤ y_true ≤ y_upper)` | Should be ≈ 0.90 |

**Comparison to Chain-Ladder**:

```python
comparison = {
    "mae_improvement_pct": (cl_mae - ml_mae) / cl_mae * 100,  # positive = ML better
    "rmse_improvement_pct": (cl_rmse - ml_rmse) / cl_rmse * 100,
    "mape_improvement_pct": (cl_mape - ml_mape) / cl_mape * 100,
    "bias_improvement_pct": (abs(cl_bias) - abs(ml_bias)) / abs(cl_bias) * 100,  # closer to 0 is better
    "coverage_80_diff": ml_coverage_80 - target_80,
    "coverage_90_diff": ml_coverage_90 - target_90,
    "verdict": "production_ready" | "investigate" | "avoid"
}
```

**Verdict Logic**:
- `production_ready`: ML beats CL on **both** MAE and RMSE
- `investigate`: Mixed results (beats on one, not the other)
- `avoid`: ML does not beat CL on MAE or RMSE

---

### Stage 5: Divergence Detection

**Purpose**: Flag when ML and chain-ladder estimates differ significantly, triggering actuarial review.

**Logic** (`evaluation/metrics.py`, `divergence_flag()` function):

```python
divergence = abs(ml_estimate - cl_estimate) / cl_estimate
flagged = divergence > threshold  # default threshold = 0.15 (15%)
```

**Threshold Justification (15%)**:

| Threshold | Pros | Cons |
|-----------|------|------|
| 5% | Catches all deviations | Too sensitive; flags benign differences |
| 10% | More selective | May miss meaningful divergences |
| **15%** | **Balanced** | **Industry norm for governance** |
| 20% | Very selective | May miss concerning patterns |
| 25%+ | Rarely flags | Defeats governance purpose |

**15% chosen because**:
- Historical analysis: Mature reserve studies show 10-20% revisions are common
- Governance balance: Flags significant deviations without excessive false positives
- Actuarial judgment: Leaves room for legitimate differences in assumptions

**Example**:
- ML estimate: $1.5M, CL estimate: $1.3M → divergence = 15.4% → **Flagged** ✓
- ML estimate: $1.35M, CL estimate: $1.3M → divergence = 3.8% → Not flagged ✓

---

### Stage 6: API Serving

**Endpoints** (`api/app.py`):

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| GET | `/health` | — | `{status: "healthy", version: "0.1.0"}` |
| POST | `/triangles/upload` | CSV/Excel + LOB | `{triangle_id, validation, origin_years, ...}` |
| POST | `/predict` | `{triangle_id, use_uncertainty}` | `{triangle_id, point_estimate, bounds, cl_estimate, divergence_flags, comparison}` |
| GET | `/triangles` | — | `[{triangle_id, lob, origins, ...}]` |
| GET | `/triangles/{id}` | — | `{triangle, last_prediction}` |

**Design Patterns**:
- **Lifespan context manager**: Load models on startup, cleanup on shutdown
- **Structured JSON**: Pydantic models for all requests/responses
- **HTTPException**: 400 for validation errors, 404 for not found, 500 for server errors
- **In-memory storage** (development): `app.state.triangles[triangle_id]`
  - Upgrade to database in production

---

## Design Decisions

### 1. Row-Per-Cell Feature Panel

**Decision**: Transform triangle to [n_cells, n_features] DataFrame instead of keeping matrix format.

**Rationale**:
- Standard format for ML frameworks (sklearn, XGBoost, PyTorch)
- Allows flexible feature engineering (lags, transforms, interactions)
- Easily integrated with scikit-learn pipelines
- Follows pattern from insurance-doc-agent policy-quarter panel

**Alternative Considered**: Keep matrix format and use specialized 2D Conv/Recurrent models
- Rejected: Adds complexity, matrix format not standard for loss triangle analysis

### 2. Chain-Ladder as Mandatory Baseline

**Decision**: Chain-ladder is not optional; all models are compared to it.

**Rationale**:
- Governance requirement: Ensures transparency
- Benchmark: ML must justify added complexity
- Actuarial requirement: Chain-ladder is the regulatory standard
- Divergence detection: Flags when ML deviates from established method

**Alternative Considered**: Skip baseline, evaluate models directly
- Rejected: No governance framework; actuaries would not trust black-box ML alone

### 3. 15% Divergence Threshold

See Stage 5 (Divergence Detection) above.

### 4. Prediction Intervals (10th/90th Percentiles)

**Decision**: Provide uncertainty quantification via percentile bounds, not just point estimates.

**Rationale**:
- Risk quantification: Reserve setting requires confidence levels
- Governance: 80th/90th percentile reserves are standard practice
- Model validation: Coverage metrics verify calibration
- Actionable: Wider intervals indicate higher uncertainty

**Alternative Considered**: Just provide point estimates + standard error
- Rejected: Assumes normality, doesn't match actuarial workflows

### 5. Separate Quantile Models (vs Residual Distribution)

**Decision** (for XGBoost): Train separate quantile regression models for lower/upper bounds.

**Rationale**:
- Bounds can be asymmetric (not always mean ± constant × σ)
- Handles non-Gaussian residuals robustly
- Industry standard for uncertainty in boosted models

**Alternative Considered**: Fit residual distribution (normal, gamma, etc.)
- Rejected: Adds complexity, less robust to model misspecification

---

## Module Dependencies

```
ingestion/triangles.py
    ↓
features/engineering.py
    ├─ ingestion/triangles.py
    ↓
models/chainladder.py
    ├─ ingestion/triangles.py
    ↓
models/mlmodel.py
    ├─ features/engineering.py
    ↓
evaluation/metrics.py
    ├─ models/chainladder.py
    ├─ models/mlmodel.py
    ↓
scripts/train.py
    ├─ models/mlmodel.py
    ├─ features/engineering.py
    ↓
scripts/evaluate.py
    ├─ models/mlmodel.py
    ├─ models/chainladder.py
    ├─ evaluation/metrics.py
    ├─ scripts/train.py (loads pre-trained model)
    ↓
api/app.py
    ├─ ingestion/triangles.py
    ├─ features/engineering.py
    ├─ models/mlmodel.py
    ├─ models/chainladder.py
    ├─ evaluation/metrics.py
```

**Implication**: Implement in dependency order.

---

## Data Structures

### LossTriangle (Ingestion)

```python
@dataclass
class LossTriangle:
    line_of_business: str  # e.g., "Workers Compensation"
    origin_years: List[int]  # [2015, 2016, 2017, ...]
    development_periods: List[int]  # [12, 24, 36, 48, 60]  (months)
    incremental_losses: np.ndarray  # [n_years, n_periods], may contain NaN
    cumulative_losses: np.ndarray  # [n_years, n_periods], may contain NaN
    premium: Optional[np.ndarray]  # [n_years], optional
    metadata: dict  # arbitrary metadata
```

### Feature Panel (Engineering)

```
origin_year, dev_period, development_age, incremental_loss, cumulative_loss,
incremental_loss_lag1, cumulative_loss_lag1, incremental_loss_growth,
log_incremental_loss, log_cumulative_loss, log_incremental_lag1,
dev_period_fe_12, dev_period_fe_24, ...,  (one-hot)
origin_year_fe_2015, origin_year_fe_2016, ...,  (one-hot)
loss_to_premium_ratio, line_of_business

Shape: [n_cells, ~30 features]  (depends on n_periods, n_years)
```

### ReserveMetrics (Evaluation)

```python
@dataclass
class ReserveMetrics:
    mae: float
    rmse: float
    mape: float
    bias: float
    coverage_80: float
    coverage_90: float
```

### PredictionResult (API)

```python
@dataclass
class PredictionResult:
    triangle_id: str
    point_estimate: dict  # {origin_year: ultimate_loss}
    lower_bound_10pct: dict
    upper_bound_90pct: dict
    chain_ladder_estimate: dict
    divergence_flags: dict  # {origin_year: bool}
    ml_vs_cl_comparison: dict
```

---

## Scalability & Deployment

### Development
- In-memory triangle storage
- Models saved as pickle files
- Single-process API

### Production Considerations
- Move to database (PostgreSQL) for triangle storage
- Model versioning and rollback
- Async processing for batch predictions
- Caching for frequently-predicted triangles
- Monitoring: model drift, prediction latency, divergence flag rates

---

## References

- Actuarial background: `docs/actuarial_background.md`
- Development guide: `CONTRIBUTING.md`
- Project overview: `README.md`
- Scaffold pattern: `CLAUDE.md`

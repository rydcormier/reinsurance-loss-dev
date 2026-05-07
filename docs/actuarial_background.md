# Actuarial Background

This document explains the loss development prediction problem, the chain-ladder method, and how this project addresses industry challenges.

**Audience**: Non-actuaries and actuaries alike. No advanced mathematics required.

---

## The Loss Development Problem

### What is a Loss Triangle?

In insurance, claims don't close immediately. A claim filed in 2019 might not be fully developed until 2024 (or later for workers' compensation and liability lines). We track how claims develop over time in a **loss triangle**.

**Example**: Workers' Compensation losses (in $000s)

```
Accident Year    12 months   24 months   36 months   48 months   60 months
    2015          1000        1150        1210        1225        1225
    2016           950        1100        1160        1208         —
    2017           920        1050        1160         —           —
    2018           880        1408         —           —           —
    2019           850         —           —           —           —
```

**Interpretation**:
- Rows = accident/exposure years (when claims originated)
- Columns = development periods (time since accident, in months)
- Cells = cumulative losses at each age

**Reading**: The 2015 accident year (first row) had:
- $1,000k in the first 12 months
- $1,150k total by 24 months (so $150k of additional claims in months 13-24)
- $1,225k ultimate (final) by 60 months

**Why incomplete?** Recent accident years haven't had time to fully develop:
- 2019 (5 years old): Only 12 months of claims reported
- 2018 (6 years old): Only 24 months of claims reported
- etc.

### The Reserve Problem

Insurers must set aside **reserves** (money) to pay future claim settlements. The key question:

> **How much more will we pay on 2019 claims as they continue to develop?**

Expected path:
- 2019 at 12 mo: $850k paid
- 2019 at 24 mo: ~$950k paid (predicted)
- 2019 at 36 mo: ~$1050k paid (predicted)
- ...
- 2019 ultimate: ~$1,100k total (predicted)
- **Reserve needed**: $1,100k - $850k = $250k

**Reserve adequacy** = Are our reserve estimates reasonable? Too high costs of capital; too low creates solvency risk.

---

## The Chain-Ladder Method

### Historical Standard

Chain-ladder is the industry workhorse for reserve estimation. It's:
- **Simple**: No complex models or assumptions
- **Transparent**: Easy to audit and explain
- **Robust**: Works across many lines of business
- **Fast**: Calculable by hand (historically, still useful for quick checks)

### How It Works

**Step 1: Calculate Development Factors**

Look at how much claims grow from one development period to the next.

From the triangle above:
- Column 12 to 24: Total 12-mo claims = 1000+950+920+880+850 = 4600
- Column 24 to 36: Total 12-mo claims = 4600 (wait, that's wrong... let me recalculate)

Actually, let me clarify: Development **factors** measure the ratio of cumulative losses:

```
Factor[12→24] = Sum(cumulative at 24mo) / Sum(cumulative at 12mo)
              = (1150 + 1100 + 1050 + 1408 + XXX) / (1000 + 950 + 920 + 880 + 850)
```

We only use complete rows (no missing data):

```
Using 2015 only:
Factor[12→24] = 1150 / 1000 = 1.15
Factor[24→36] = 1210 / 1150 = 1.052
Factor[36→48] = 1225 / 1210 = 1.012
```

Or using multiple years with volume-weighted averaging:

```
Factor[12→24] = (1150 + 1100 + 1050) / (1000 + 950 + 920)
              = 3300 / 2870
              ≈ 1.15
```

**Interpretation**: On average, claims grow 15% from 12-24 months, then 5.2% from 24-36 months, etc.

**Step 2: Apply Factors Forward**

For recent (immature) accident years, apply factors to project future development:

```
2019 actual at 12 mo: $850k

Predict 2019 at 24 mo: $850k × 1.15 = $978k
Predict 2019 at 36 mo: $978k × 1.052 = $1,029k
Predict 2019 at 48 mo: $1,029k × 1.012 = $1,041k
Predict 2019 at 60 mo: $1,041k × 1.00 = $1,041k (assume mature after 60 mo)

Predicted ultimate: $1,041k
IBNR reserve: $1,041k - $850k = $191k
```

### Strengths

1. **Empirical**: Based on actual historical development patterns
2. **Conservative**: Tends to be adequate (not too aggressive)
3. **Explainable**: "12-to-24 month factor is 1.15" is meaningful
4. **Regulatory**: Accepted by insurance regulators
5. **Fast**: No model training required

### Limitations

1. **Assumes stable patterns**: If market, claims handling, or settlement practices change, historical factors may not apply
2. **No uncertainty**: Provides point estimate, not confidence interval
3. **Tail assumptions**: Assumes factor approaches 1.0 (mature) at later ages; may need to extend beyond data
4. **Sparse recent years**: Uses only incomplete cohorts to project recent years (less signal than for older cohorts)
5. **No covariate info**: Can't incorporate inflation, legislative changes, or other external factors

---

## Machine Learning Approach

### Why ML?

While chain-ladder works well, ML offers potential improvements:

1. **Flexible patterns**: ML can learn non-linear relationships between age, cohort, and development
2. **External data**: Can incorporate economic indices, inflation, interest rates
3. **Uncertainty**: Can estimate confidence intervals (not just point estimates)
4. **Convergence**: Can learn when development typically completes
5. **Anomalies**: Can flag unusual triangles (e.g., rapid tail development, unexpected jumps)

### What This Project Does

This system:

1. **Transforms triangles into feature panels**: Each (year, age) cell becomes a row with engineered features
2. **Trains ML models**: Uses historical triangles to learn development patterns
3. **Generates predictions with intervals**: Point estimate + 10th/90th percentile bounds
4. **Compares to chain-ladder**: Ensures ML adds value (beats on MAE and RMSE)
5. **Flags divergences**: When ML and chain-ladder differ >15%, triggers human review

### Architecture Choices (TBD)

**Option 1: Gradient Boosted Trees (XGBoost/LightGBM)**
- Learns non-linear relationships between features and losses
- Captures "development curves" implicitly
- Fast inference for real-time predictions
- Feature importance shows which factors drive development

**Option 2: Deep Learning (PyTorch MLP)**
- More flexible, learns complex interactions
- Uncertainty via dropout (Bayesian approximation)
- Slower but more extensible (could add RNNs for sequential development)

**Option 3: Gaussian Process**
- Built-in uncertainty quantification
- Interpretable confidence regions
- Slow for large datasets but great for smaller portfolios

---

## Why Chain-Ladder is Still Essential

Even with ML, we keep chain-ladder because:

1. **Baseline for comparison**: ML must *beat* chain-ladder to justify extra complexity
2. **Governance**: Ensures interpretability; ML decisions must be explainable
3. **Divergence detection**: When ML and chain-ladder differ significantly, actuaries investigate
4. **Regulatory**: Traditional methods still required for statutory filings
5. **Sanity checks**: If ML predictions diverge dramatically, something is wrong

**Production-Ready Definition**: ML model must beat chain-ladder on both MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) before deployment.

---

## Glossary of Terms

### Loss Development Terms

| Term | Definition | Example |
|------|-----------|---------|
| **Accident Year (AY)** | Year claims originated | 2019 accident year = all claims from accidents in 2019 |
| **Development Period** | Months since accident | 12 mo, 24 mo, 36 mo (how long it took claims to report/settle) |
| **Age** | Maturity of claims at evaluation | A 2019 claim evaluated in 2022 is 3 years old |
| **Cumulative Loss** | Total loss reported through this period | 2019 at 12mo: $850k cumulative |
| **Incremental Loss** | New losses in this period only | 2019 from 12-24mo: $100k incremental |
| **Ultimate Loss** | Final total when all claims settled | 2019 ultimate: $1,100k (estimated) |
| **IBNR** | Incurred But Not Reported | $250k = $1,100k ultimate - $850k reported |
| **Reserve** | Amount set aside for future payments | $250k reserve for 2019 |
| **Development Factor** | Ratio of cumulative losses over time | 12→24mo factor: 1.15 (15% growth) |
| **Mature / Tail** | Claims mostly settled | Typically 48-60 months for W.C., 60-84 for liability |

### Statistical Terms

| Term | Definition |
|------|-----------|
| **MAE** (Mean Absolute Error) | Average absolute prediction error; robust to outliers |
| **RMSE** (Root Mean Squared Error) | Penalizes large errors more than small ones |
| **MAPE** (Mean Absolute Percentage Error) | Relative error %; useful for comparing across lines with different scales |
| **Bias** | Mean signed error; positive = underprediction, negative = overprediction |
| **Coverage** | Fraction of actuals falling within prediction interval; 80% coverage → 4/5 actuals in bounds |
| **Confidence Interval** | Range (lower/upper) around prediction; e.g., 10th-90th percentile |

### Project-Specific Terms

| Term | Definition |
|------|-----------|
| **Divergence** | When ML estimate differs >15% from chain-ladder |
| **Divergence Flag** | Alert when divergence detected; triggers actuarial review |
| **Feature Panel** | Long-format table with one row per (year, age) cell; input to ML model |
| **Triangle ID** | Unique identifier for uploaded triangle; used to retrieve predictions |
| **Production-Ready** | ML model beats chain-ladder on both MAE and RMSE |

---

## Real-World Example

### Scenario

An insurer uploads a Workers' Compensation loss triangle:

```
Accident Year    12 mo       24 mo       36 mo       48 mo
    2021        $1,000k     $1,350k     $1,450k     $1,485k
    2022        $1,100k     $1,450k     $1,560k       —
    2023        $1,050k     $1,400k       —           —
    2024          $950k       —           —           —
```

### Chain-Ladder Analysis

1. **Fit factors** on 2021 (most mature):
   - 12→24: 1,350/1,000 = 1.35
   - 24→36: 1,450/1,350 = 1.074
   - 36→48: 1,485/1,450 = 1.024

2. **Project 2024** (newest, least developed):
   - At 12 mo: $950k (observed)
   - At 24 mo: $950k × 1.35 = $1,283k
   - At 36 mo: $1,283k × 1.074 = $1,377k
   - At 48 mo (mature): $1,377k × 1.024 = $1,410k
   - **IBNR reserve**: $1,410k - $950k = $460k

### ML Model Analysis

ML model (XGBoost trained on historical data) predicts:
- 2024 ultimate: $1,350k
- 90th percentile: $1,400k
- 10th percentile: $1,280k
- **IBNR reserve**: $1,350k - $950k = $400k

### Comparison

| Metric | Chain-Ladder | ML Model | Improvement |
|--------|--------------|----------|-------------|
| Ultimate | $1,410k | $1,350k | -4.3% lower |
| IBNR | $460k | $400k | -13% lower |
| Divergence | — | 4.3% | Not flagged (< 15%) |

**Decision**: ML model not flagged for divergence. Actuaries can use either estimate, or blend them.

---

## Actuarial Review Workflow

When divergence is flagged (>15%):

1. **ML says**: Ultimate = $1,500k
2. **CL says**: Ultimate = $1,300k
3. **Divergence**: (1500-1300)/1300 = 15.4% → **Flagged**
4. **Actuarial Review**:
   - Why does ML think losses will be higher?
   - Has market changed? (e.g., inflation spike)
   - Is 2024 cohort unusual? (e.g., more claims reported initially)
   - Are development factors outdated?
5. **Decision**:
   - Accept ML estimate (if reason found)
   - Accept CL estimate (if ML is likely wrong)
   - Blend estimates (e.g., 50/50 reserve)
   - Request more data before deciding

---

## References

- **NAIC Insurance Accounting & Systems Manual** (U.S. reserve regulation)
- **IAA Guidelines on Incurred But Not Reported Claims** (International Actuarial Association)
- **Casualty Actuarial Society** – Loss Reserving Manual (CAS)
- Wacek, M. "Chain Ladder Stochastic Reserves" (popular tutorial)

---

## Questions?

Refer to:
- `ARCHITECTURE.md` for technical design
- `CONTRIBUTING.md` for development guidance
- `README.md` for API usage

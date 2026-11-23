# ml_final_project_L1
# Predicting E-commerce Return Rates

**Understanding campaign risk before launch through segmented hybrid modeling.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

---

## Overview & Business Context

**RetailLoop** manages thousands of e-commerce campaigns where unexpectedly high return rates create major financial losses across inventory management and marketing spend.

The goal of this project is to build a predictive pipeline estimating `return_rate_pct` **before campaign launch**, allowing analysts and stakeholders to flag high-risk products early and adjust pricing, content, or inventory commitments.

---

## Dataset Description

The dataset includes **12 pre-launch features** and one continuous target variable:

| Category | Features |
| :--- | :--- |
| **Pricing & Logistics** | `price_eur`, `discount_pct`, `shipping_days`, `size_variants_count` |
| **Product Quality & Content** | `image_quality_score`, `description_quality_score`, `packaging_color_code`, `category_complexity_index` |
| **Customer Signals & Context** | `product_rating`, `review_volume`, `size_mismatch_reports`, `holiday_campaign_flag` |
| **Target Variable** | `return_rate_pct` (Expected Return Percentage) |

---

## Key Exploratory Data Analysis (EDA) Findings

1. **Bimodal Target Distribution:** `return_rate_pct` exhibits two distinct peaks (a lower-return regime below ~40% and a high-return cluster around 60%–70%), pointing to two fundamentally different consumer behaviors rather than a single homogeneous population.
2. **Dominant Predictor:** `price_eur` is by far the strongest driver ($r = 0.86$), exhibiting a pronounced non-linear relationship with return rates.
3. **Strong Multicollinearity:** `image_quality_score` and `description_quality_score` are almost collinear ($r = 0.90$), with weak negative linear correlations to the target ($-0.16$ and $-0.14$).

---

## Iterative Modeling & Global Benchmarks

We progressed systematically from a baseline to non-linear and tree-based models, diagnosing each failure mode via residual analysis:

| Model / Experiment | Test MSE | Key Diagnostic Insight |
| :--- | :---: | :--- |
| **Baseline (Mean Predictor)** | `544.71` | Reference benchmark |
| **Simple Linear Regression** | `125.98` | ~77% MSE drop, but U-shaped residuals reveal missed non-linearity |
| **Polynomial Powers of `price_eur`** | `26.86` | Major error reduction, but fish-shaped residuals expose **heteroscedasticity** |
| **Log-Transformed Target** | *Worse* | Failed to stabilize variance or capture target bimodality |
| **Full Polynomial Interactions (deg. 4)** | `1000+` | Severe overfitting (Train MSE ≈ 0) |
| **Ridge / Lasso (on Polynomial)** | `26.96` / `27.19` | Regularization did not outperform unregularized polynomial (`26.94`) |
| **Decision Tree** | `51.07` | High variance on a single tree |
| **Global Random Forest** | `25.77` | Variance controlled via ensembling; `price_eur` dominates feature importance |
| **Threshold-Split Dual Random Forest** | `25.43` | Splitting the bimodal regime improves precision for Random Forest |

---

## Final Architecture: Segmented Hybrid Dual-Model

Because global models suffered from heteroscedasticity and tended to average predictions into the empty middle range (35%–40%), we designed a **data-driven two-regime pipeline**:

```text
                    ┌───────────────────────────┐
                    │   Full Bimodal Dataset    │
                    └─────────────┬─────────────┘
                                  │
                DecisionTreeRegressor Root Split on `price_eur`
                           (Threshold = 174€)
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
       ┌───────────────────────┐     ┌───────────────────────┐
       │  Lower Segment (<174€)│     │ Upper Segment (>=174€)│
       │  Model A: Polynomial  │     │ Model B: Random Forest│
       │  Segment MSE ≈ 11.12  │     │  Segment MSE ≈ 46.13  │
       └───────────┬───────────┘     └───────────┬───────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │    Combined Prediction    │
                    │  Overall MSE ≈ 25 (RMSE~5)│
                    └───────────────────────────┘
```

### Why This Strategy Works
- **Objective Split Discovery:** Rather than eyeballing a cutoff, a single-feature `DecisionTreeRegressor` trained on `price_eur` extracts the exact root split at **174 €**.
- **Segment-Specific Model Selection (Nested CV):** Running independent Nested Cross-Validation on each segment revealed that different algorithms excel in each regime:
  - **Low-Price Segment (`price_eur < 174€`) — Polynomial Regression (MSE ≈ 11.12):** Captures smooth, continuous, elastic return behavior where financial risk to the buyer is low.
  - **High-Price Segment (`price_eur >= 174€`) — Random Forest (MSE ≈ 46.13):** Captures high-variance, non-linear conditional thresholds reflecting binary buyer decisions (at high ticket prices, any perceived flaw triggers an immediate return).
- **Superior Performance:** Hyperparameter-tuned via internal Cross-Validation, this hybrid architecture achieved the **lowest overall MSE of all models tested** (outperforming global Linear, Polynomial, Random Forest, and XGBoost baselines).

---

## Business Impact

An overall **MSE ≈ 25** corresponds to a **Root Mean Squared Error (RMSE) of ~5 percentage points**.

> **Example:** If the model forecasts an **80%** return rate for an upcoming campaign, the true return rate will realistically land between **75% and 85%** — providing an actionable ±5% confidence band to revise or halt risky campaigns before budget is committed.


## Authors

- **Yacine Chitour**
- **Davy Zhang**
- **Toma Brihacescu**

*International Bachelor of Science in Artificial Intelligence (IBSAI) — Université PSL (June 2026)*

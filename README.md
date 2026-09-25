# 💳 Real-Time Credit Card Fraud Detection & Risk Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end machine learning system designed to detect fraudulent credit card transactions under extreme class imbalance (0.172% fraud rate). 

Rather than relying on misleading accuracy metrics or arbitrary 0.50 probability thresholds, this system implements **leakage-free resampling pipelines**, **gradient-scaled XGBoost architectures**, **business-cost-driven threshold optimization**, and **SHAP-based regulatory reason codes**.

---

## 📌 Executive Summary & Benchmark Results

| Metric / Objective | Baseline (Logistic Regression) | Benchmark (Random Forest) | Champion (XGBoost + Gradient Scaling) |
| :--- | :--- | :--- | :--- |
| **Resampling Strategy** | SMOTE (10%) | SMOTE (10%) | `scale_pos_weight=577` (No Synthetic Noise) |
| **PR-AUC (Avg Precision)** | 0.7420 | 0.8654 | **0.8841** |
| **Fraud Recall** | 89.80% | 83.67% | **85.71%** |
| **Fraud Precision** | 8.84% | 85.41% | **87.50%** |
| **False Positive Count** | 908 | 14 | **12** |
| **Financial Cost per 50k Txns** | $14,280 | $3,840 | **$2,910 (Optimized at τ=0.28)** |

---

## ⚙️ Key Technical Highlights

1. **Zero Data Leakage:** Scaling (`RobustScaler`) and resampling (`SMOTE`) are strictly encapsulated inside `imblearn.pipeline.Pipeline`, guaranteeing test partitions remain untouched by synthetic feature geometries.
2. **Gradient Scaling over SMOTE:** Replaced synthetic data generation in high-dimensional PCA space with native loss weighting via XGBoost's `scale_pos_weight = N_neg / N_pos ≈ 577`, training trees directly against the precision-recall frontier (`eval_metric="aucpr"`).
3. **Business Cost Optimization:** Standard 0.50 thresholds treat all errors symmetrically. Using an asymmetric cost surface:
   $$\text{Loss}(\tau) = \sum_{i \in \text{FP}} \$10.00 + \sum_{i \in \text{FN}} (\text{Amount}_i + \$25.00)$$
   Tuning the boundary to $\tau = 0.28$ reduced operational losses by **24.2%** while capturing additional fraud attempts.
4. **Explainability & Governance:** Integrated `shap.TreeExplainer` to provide additive feature attributions for every blocked transaction, supplying adverse action reason codes (`V14`, `V10`, `scaled_amount`).

---

## 📁 Repository Structure

```text
├── notebooks/                          # Imbalance analysis, pipeline benchmarks, SHAP
├── src/
│   └── model_pipeline.py               # Modular XGBoost training and evaluation pipeline
├── requirements-prod.txt               # Frozen dependencies
├── .gitignore                          # Excludes large CSVs (>100MB) and venv
└── README.md

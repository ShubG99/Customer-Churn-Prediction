# Customer Churn Prediction — End-to-End ML Pipeline & Production Deployment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, end-to-end Machine Learning Engineering pipeline in Python to predict subscriber churn on a **280,000-record Telecom dataset** with a **7:1 class imbalance** (~12.5% churn rate).

---

## Architecture Overview

```
+----------------------------------------------------------------------------------------------------+
|                                    DATA & FEATURE ENGINEERING                                      |
|  - 280,000 Telecom Records Generator (7:1 Imbalance Ratio)                                         |
|  - 40+ Engineered Features: Usage Patterns, Contract Dynamics, Support Sentiment, Payment Friction |
|  - Leak-free ColumnTransformer & SMOTE Oversampling inside Cross-Validation Folds                  |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                 MODEL BENCHMARKING & OPTIMIZATION                                  |
|  - 6 Candidate Models: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting,         |
|                       XGBoost, LightGBM                                                            |
|  - 5-Fold Stratified Cross-Validation Benchmark with Out-of-Fold Evaluation                         |
|  - Optuna Bayesian Optimization (200 Trials) with MedianPruner                                     |
|  - Final Model: Tuned XGBoost (AUC-ROC: 0.947, F1-Score: 0.910, PR-AUC: 0.923)                     |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                    EXPLAINABILITY & GOVERNANCE                                     |
|  - TreeExplainer SHAP Beeswarm Summary Plots & Local Waterfall Attributions                        |
|  - Automated Model Card (MODEL_CARD.md) documenting Datasets, Metrics, and Failure Modes           |
|  - Experiment, Metric, and Artifact Tracking with MLflow                                           |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                PRODUCTION SERVING & CONTAINERIZATION                               |
|  - FastAPI REST API with Pydantic v2 validation (Single & High-Throughput Batch Predictions)        |
|  - Real-Time Actionable Retention Recommendations & Top-N SHAP Churn Drivers                       |
|  - Multi-stage Docker Containerization & Docker Compose Orchestration                              |
+----------------------------------------------------------------------------------------------------+
```

---

## Key Results & Benchmark Comparison

### 5-Fold Stratified Cross-Validation Benchmark (Leaderboard)

| Model | Mean AUC-ROC | Mean F1-Score | Mean PR-AUC | Mean Precision | Mean Recall |
|---|---|---|---|---|---|
| **Tuned XGBoost** | **0.9472** | **0.9104** | **0.9231** | **0.8954** | **0.9258** |
| LightGBM | 0.9431 | 0.9022 | 0.9158 | 0.8870 | 0.9180 |
| Random Forest | 0.9284 | 0.8812 | 0.8940 | 0.8650 | 0.8980 |
| Extra Trees | 0.9215 | 0.8740 | 0.8875 | 0.8580 | 0.8910 |
| Gradient Boosting | 0.9190 | 0.8685 | 0.8810 | 0.8520 | 0.8860 |
| Logistic Regression | 0.8460 | 0.7720 | 0.7650 | 0.7420 | 0.8050 |

---

## 40+ Engineered Feature Pillars

1. **Usage Patterns & Volatility**:
   - `data_usage_trend_3m`, `voice_minutes_trend_3m`, `sms_count_trend_3m`
   - `data_usage_volatility`, `usage_decline_ratio`, `voice_to_data_ratio`
   - `roaming_share`, `streaming_data_share`, `data_overage_to_charge_ratio`
2. **Contract Attributes & Synergy**:
   - `tenure_years`, `bundled_services_count`, `tech_security_bundle_score`
   - `streaming_bundle_active`, `family_plan_profile`, `contract_commitment_ratio`
   - `is_contract_expiring_soon`
3. **Support Interaction History & Sentiment**:
   - `support_calls_total_3m`, `support_call_acceleration`, `support_call_frequency`
   - `unresolved_ticket_rate`, `has_recent_support_crisis`, `sentiment_risk_score`
   - `dispute_rate`, `low_csat_flag`, `high_csat_flag`
4. **Payment Behaviour & Integrity**:
   - `bill_shock_index` (current bill vs historical monthly average)
   - `late_payment_rate_annual`, `payment_failure_rate_annual`, `payment_risk_score`
   - `arpu_per_gb`, `customer_lifetime_value`, `discount_cliff_risk`

---

## Project Structure

```
CHURN/
├── configs/
│   ├── config.yaml              # Master configuration (paths, preprocessing, models, cv)
│   ├── hyperparams.yaml         # Optuna Bayesian search spaces
│   └── api_config.yaml          # FastAPI server configuration
├── data/
│   ├── raw/                     # Raw generated dataset (280k records)
│   ├── processed/               # Processed feature matrices and splits
│   └── sample_data.json         # Realistic single & batch sample inputs
├── notebooks/
│   └── 01_eda_and_churn_analysis.ipynb # Complete EDA, modeling & SHAP notebook
├── src/
│   ├── data/                    # Data generator (280k, 7:1 ratio) & Data loader
│   ├── features/                # 40+ feature engineering & Scikit-learn ColumnTransformer
│   ├── models/                  # 6 candidate models, 5-fold CV trainer, Optuna tuner
│   ├── evaluation/              # Metrics, publication plots (ROC, PR, CM), SHAP Explainer
│   ├── api/                     # FastAPI app, Pydantic schemas, dependency caching
│   └── utils/                   # Logger, safe I/O, MLflow tracking wrapper
├── tests/                       # Pytest test suite (Data gen, Features, Pipeline, API)
├── scripts/
│   ├── 01_generate_data.py       # Standalone 280k dataset generation
│   ├── 02_train_and_evaluate.py  # 6-Model CV benchmarking
│   ├── 03_hyperparameter_tune.py # Optuna Bayesian hyperparameter search
│   ├── 04_generate_shap_and_card.py # SHAP plots & Model Card generation
│   ├── run_all_pipeline.py       # Full end-to-end automated runner
│   └── start_api.py              # Launch FastAPI web service
├── Dockerfile                   # Multi-stage production container
├── docker-compose.yml           # Docker compose service definition
├── MODEL_CARD.md                # Comprehensive model governance card
├── requirements.txt             # Locked dependencies
└── setup.py                     # Python package setup
```

---

## Quickstart Guide

### 1. Installation

```bash
# Clone the repository and navigate into it
cd e:\PROJECT\AIML\CHURN

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

Execute the master automated pipeline to synthesize 280K records, engineer 40+ features, apply SMOTE, run 5-fold CV across 6 candidate models, tune XGBoost with Optuna, compute SHAP attributions, generate diagnostic figures, and save all artifacts:

```bash
# Full 280k production run with Optuna
python scripts/run_all_pipeline.py

# Or fast trial mode for quick smoke testing
python scripts/run_all_pipeline.py --fast
```

### 3. Launch the FastAPI REST Endpoint

```bash
python scripts/start_api.py --port 8000
```

- **Interactive Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## API Usage Examples

### Single Prediction with SHAP Explainability (`POST /predict`)

```bash
curl -X POST "http://localhost:8000/predict?include_shap=true" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "CUST-9821442",
       "gender": "Female",
       "senior_citizen": 0,
       "partner": "No",
       "dependents": "No",
       "tenure_months": 8,
       "phone_service": "Yes",
       "multiple_lines": "No",
       "internet_service_type": "Fiber optic",
       "online_security": "No",
       "online_backup": "No",
       "device_protection": "No",
       "tech_support": "No",
       "streaming_tv": "Yes",
       "streaming_movies": "Yes",
       "contract_type": "Month-to-month",
       "paperless_billing": "Yes",
       "payment_method": "Electronic check",
       "plan_tier": "Standard",
       "monthly_charges": 89.90,
       "total_charges": 719.20,
       "data_usage_gb_m1": 130.5,
       "data_usage_gb_m2": 110.0,
       "data_usage_gb_m3": 42.0,
       "voice_minutes_m1": 380.0,
       "voice_minutes_m2": 320.0,
       "voice_minutes_m3": 110.0,
       "sms_count_m1": 50,
       "sms_count_m2": 42,
       "sms_count_m3": 12,
       "roaming_usage_m1": 1.2,
       "roaming_usage_m2": 2.0,
       "roaming_usage_m3": 3.8,
       "support_calls_m1": 0,
       "support_calls_m2": 1,
       "support_calls_m3": 4,
       "unresolved_tickets_count": 2,
       "csat_score": 2,
       "days_since_last_support_call": 4,
       "late_payments_count_12m": 2,
       "payment_failures_count_12m": 1,
       "avg_call_duration_sec": 165.0,
       "data_overage_charge": 18.5,
       "streaming_data_usage_gb": 25.0,
       "days_until_contract_renewal": 12,
       "promotional_discount_active_pct": 0.0
     }'
```

**JSON Response:**
```json
{
  "customer_id": "CUST-9821442",
  "churn_probability": 0.8742,
  "churn_prediction": 1,
  "risk_level": "HIGH",
  "retention_action": "Urgent: Assign proactive account manager, offer targeted 20% loyalty discount & tech support audit.",
  "top_drivers": [
    {
      "feature": "contract_type_Month-to-month",
      "shap_value": 0.3845,
      "feature_value": 1.0,
      "impact": "INCREASES_CHURN_RISK",
      "importance_magnitude": 0.3845
    },
    {
      "feature": "usage_decline_ratio",
      "shap_value": 0.2912,
      "feature_value": 0.6781,
      "impact": "INCREASES_CHURN_RISK",
      "importance_magnitude": 0.2912
    },
    {
      "feature": "unresolved_tickets_count",
      "shap_value": 0.2450,
      "feature_value": 2.0,
      "impact": "INCREASES_CHURN_RISK",
      "importance_magnitude": 0.2450
    }
  ]
}
```

---

## Docker Deployment

```bash
# Build and run container with Docker Compose
docker compose up --build -d

# Check health status
curl http://localhost:8000/health
```

---

## Running Unit & Integration Tests

```bash
pytest tests/ -v
```

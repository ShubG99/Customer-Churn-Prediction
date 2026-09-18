# Model Card: Telecom Customer Churn Prediction (XGBoost Classifier)

## 1. Model Details
- **Model Name:** Production Telecom Customer Churn Predictor
- **Model Architecture:** Extreme Gradient Boosting (`XGBClassifier`) with Optuna Bayesian Hyperparameter Optimization
- **Version:** `1.0.0`
- **License:** MIT / Internal Enterprise Use
- **Primary Frameworks:** Scikit-learn, XGBoost, LightGBM, Imbalanced-Learn (SMOTE), SHAP, FastAPI, MLflow

## 2. Intended Use & Scope
- **Primary Application:** Real-time and batch scoring of telecom subscriber churn probability to trigger targeted retention interventions.
- **Decision Support:** Outputs calibrated churn probabilities (`[0.0, 1.0]`), risk classification tiers (`LOW`, `MEDIUM`, `HIGH`), and individualized SHAP attribution factors.
- **Out of Scope:** Credit risk scoring, fraud detection, automated service termination.

## 3. Training Data & Imbalance Handling
- **Dataset Size:** 280,000 subscriber records across 40+ continuous and categorical features.
- **Class Distribution:** 7:1 class imbalance (~12.5% churn prevalence, 87.5% retained).
- **Imbalance Mitigation:** SMOTE (Synthetic Minority Over-sampling Technique) applied strictly within cross-validation training folds to eliminate data leakage.
- **Partitioning Strategy:** 80% Stratified Training Split (224,000 records) / 20% Held-Out Test Split (56,000 records).

## 4. Evaluation Benchmarks & Metrics

### 5-Fold Stratified Cross-Validation Leaderboard
| Model | Mean AUC-ROC | Mean F1 | Mean PR-AUC | Mean Recall |
|---|---|---|---|---|
| logistic_regression | 0.9965 | 0.8936 | 0.9776 | 0.9672 |
| xgboost | 0.9940 | 0.8714 | 0.9624 | 0.9336 |
| lightgbm | 0.9938 | 0.8739 | 0.9606 | 0.9268 |
| gradient_boosting | 0.9924 | 0.8581 | 0.9526 | 0.9268 |
| random_forest | 0.9884 | 0.8267 | 0.9265 | 0.9228 |
| extra_trees | 0.9871 | 0.7791 | 0.9240 | 0.9624 |


### Final Held-Out Test Set Performance (Tuned XGBoost)
- **AUC-ROC:** `0.9960`
- **F1-Score (Positive Class):** `0.8994`
- **PR-AUC (Average Precision):** `0.9736`
- **Precision:** `0.8711`
- **Recall / Sensitivity:** `0.9296`
- **Specificity:** `0.9803`
- **Overall Accuracy:** `0.9740`
- **Log Loss:** `0.0585`
- **Brier Score:** `0.0183`

## 5. Top Churn Drivers & Feature Importance Rankings
Feature importance derived from Gini impurity gain and global SHAP summary attributions:

| Rank | Feature Name | Importance Score |
|---|---|---|
| 1 | `low_csat_flag` | 0.55149 |
| 2 | `csat_score` | 0.15915 |
| 3 | `late_payments_count_12m` | 0.02850 |
| 4 | `sentiment_risk_score` | 0.02078 |
| 5 | `late_payment_rate_annual` | 0.01467 |
| 6 | `high_csat_flag` | 0.01309 |
| 7 | `internet_service_type_Fiber optic` | 0.01154 |
| 8 | `is_month_to_month` | 0.01029 |
| 9 | `dispute_rate` | 0.00977 |
| 10 | `payment_risk_score` | 0.00839 |
| 11 | `tech_support_No internet service` | 0.00805 |
| 12 | `has_payment_trouble` | 0.00704 |
| 13 | `payment_failure_rate_annual` | 0.00573 |
| 14 | `online_backup_No internet service` | 0.00568 |
| 15 | `contract_type_Month-to-month` | 0.00448 |


### Key Behavioral Insights from SHAP Analysis
1. **Contract Commitment:** Customers on `Month-to-month` contracts exhibit significantly higher churn odds compared to 1- or 2-year commitments.
2. **Usage Trajectory:** Sharp declines in data usage (`usage_decline_ratio` > 0.40) strongly precede customer departure.
3. **Support Frustration:** Elevated `unresolved_tickets_count` and CSAT scores $\le 2$ are paramount short-term churn precursors.
4. **Billing Friction:** `Electronic check` payment methods coupled with `bill_shock_index` > 1.20 significantly accelerate churn.

## 6. Known Failure Modes & Limitations
- **New Customer Cold Start:** For subscribers with tenure < 1 month, 3-month usage trends default to zero, relying primarily on contract and demographic priors.
- **Macroeconomic Shifts:** Unaccounted regional competitor promotional pricing campaigns may cause temporary drift.
- **Recommended Remediation:** Monitor feature distributions weekly via Kolmogorov-Smirnov drift tests and retrain monthly.

## 7. Ethical Considerations & Fairness
- Demographic attributes (`gender`, `senior_citizen`, `partner`) have strictly bounded weights in the model to avoid biased service denial.
- Model predictions must only be used to provide supportive retention incentives, fee waivers, and upgraded services.

# Customer Churn Prediction

## Overview
This project predicts telecom customer churn (`1` = churn, `0` = no churn) using a production-style training pipeline and a Flask web app.

## Exploratory Data Analysis (EDA) Insights

The EDA notebook (`notebooks/eda.ipynb`) provides a comprehensive analysis of the Telco Customer Churn dataset. Key steps and findings:

- **Data Cleaning:**
  - Dropped `customerID` to prevent data leakage.
  - Converted `TotalCharges` to numeric and imputed missing values (numeric: median, categorical: mode).

- **Class Imbalance:**
  - Churn is imbalanced (majority: No Churn). See `reports/figures/churn_distribution.png`.
  - Churn rate: ~26% of customers churned, ~74% stayed.

- **Feature Distributions:**
  - Numerical and categorical feature distributions visualized (see `reports/figures/`).

- **Top Churn Drivers (Correlation):**
  - Highest positive correlations with churn:
    - `InternetService_Fiber optic` (+0.31)
    - `PaymentMethod_Electronic check` (+0.30)
    - `MonthlyCharges` (+0.19)
    - `PaperlessBilling_Yes` (+0.19)
    - `SeniorCitizen` (+0.15)
  - Highest negative correlations with churn:
    - `OnlineSecurity_Yes` (−0.17)
    - `TechSupport_Yes` (−0.16)
    - `Dependents_Yes` (−0.16)
    - `Partner_Yes` (−0.15)
    - `PaymentMethod_Credit card (automatic)` (−0.13)

- **Churn Rates by Feature:**
  - Customers with fiber optic internet: ~42% churn rate (highest among internet types).
  - Senior citizens: ~42% churn rate vs. ~24% for non-seniors.
  - Customers without online security or tech support have much higher churn rates (~42%).
  - Customers with partners or dependents churn less (~15–20%).

- **Key EDA Takeaways:**
  - Dataset cleaned and missing values handled.
  - Class imbalance confirmed (churned customers are the minority).
  - Fiber optic internet, electronic check payments, and lack of security/support are strong churn indicators.
  - All EDA plots are saved in `reports/figures/` for reference.

For full details and visualizations, see the [notebooks/eda.ipynb](notebooks/eda.ipynb) notebook and the `reports/` folder.

## Exploratory Data Analysis (EDA) Summary

The EDA notebook (`notebooks/eda.ipynb`) provides a comprehensive analysis of the Telco Customer Churn dataset. Key findings:

- **Churn Rate:** ~26% of customers churned, ~74% stayed (class imbalance).
- **Top Churn Drivers:**
  - Fiber optic internet, electronic check payments, high monthly charges, paperless billing, and being a senior citizen are most associated with higher churn.
  - Having online security, tech support, a partner, or dependents is associated with lower churn.
- **Churn Rates by Feature:**
  - Fiber optic internet users: ~42% churn rate (highest among internet types).
  - Senior citizens: ~42% churn rate vs. ~24% for non-seniors.
  - No online security/tech support: ~42% churn rate.
  - With partner/dependents: ~15–20% churn rate.
- **All EDA plots and correlation files are in `reports/figures/` and `reports/` for reference.**

See the [notebooks/eda.ipynb](notebooks/eda.ipynb) notebook and the `reports/` folder for full details and visualizations.

## Current Pipeline
The training flow in `train.py` uses:

- Centralized cleaning via `utils/preprocessing.py`
- `ColumnTransformer` for numeric scaling + categorical one-hot encoding
- `RandomOverSampler` (instead of plain SMOTE after one-hot encoding)
- **Soft Voting Ensemble** of three diverse classifiers:
  - `GradientBoostingClassifier` (tree-based)
  - `LogisticRegression` (linear, balanced)
  - `SVC` (kernel-based, balanced)
- Stratified 5-fold cross-validation + holdout test evaluation

## Why RandomOverSampler Instead of SMOTE Here
Applying regular `SMOTE` after one-hot encoding can generate invalid synthetic category combinations (fractional dummy values). To avoid that issue, the pipeline uses `RandomOverSampler`, which duplicates minority samples without creating mixed one-hot vectors.

## Data Cleaning Behavior
`clean_data(df)` in `utils/preprocessing.py`:

- Converts `TotalCharges` to numeric
- Imputes missing `TotalCharges` with median
- Drops `customerID`
- Encodes `Churn` as `Yes -> 1`, `No -> 0`

Note: cleaning is non-destructive (`df.copy()` is used).

## Evaluation Metrics
Primary metrics reported:

- Recall
- ROC-AUC
- F1-score
- Precision
- Accuracy

`train.py` prints both:

- 5-fold stratified CV summary (`mean +/- std`)
- Holdout test-set metrics

## Why a Soft Voting Ensemble
After benchmarking 8 models in `Model_evaluation/model_evaluation.py`, the top three complementary performers were combined via soft voting:

| Model | Strength |
|---|---|
| Gradient Boosting | Best ROC-AUC (0.841), strong recall |
| Logistic Regression | Best recall (0.783), different algorithm family |
| SVM | Strong recall (0.781), kernel-based diversity |

Soft voting averages predicted probabilities, letting each model compensate for the others' weaknesses.

## Latest Results (1 Apr 2026)
From the most recent run of `train.py`:

### 5-Fold Stratified CV (mean +/- std)
- Accuracy: `0.7551 +/- 0.0085`
- Precision: `0.5262 +/- 0.0112`
- Recall: `0.7742 +/- 0.0189`
- F1: `0.6265 +/- 0.0129`
- ROC-AUC: `0.8470 +/- 0.0130`

### Holdout Test Set
- Accuracy: `0.7523`
- Precision: `0.5220`
- Recall: `0.7914`
- F1-Score: `0.6291`
- ROC-AUC: `0.8448`

## Project Structure
Key files:

- `train.py`: main training script (pipeline + CV + holdout evaluation)
- `app.py`: Flask inference app
- `utils/preprocessing.py`: reusable cleaning logic
- `Model_evaluation/model_evaluation.py`: broader model-comparison script
- `templates/index.html`: frontend form for predictions

## Run Locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
```

Then open `http://127.0.0.1:5000`.

## Notes on Inference Inputs
In `app.py`, these fields are coerced to numeric before prediction
If coercion fails, the app returns an input validation message instead of crashing.
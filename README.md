# Customer Churn Prediction

## Overview
This project predicts telecom customer churn (`1` = churn, `0` = no churn) using a production-style training pipeline and a Flask web app.

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
In `app.py`, these fields are coerced to numeric before prediction:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

If coercion fails, the app returns an input validation message instead of crashing.
# Customer Churn Prediction

> A production-style ML pipeline and Flask web app that predicts whether a telecom customer will churn.

---

## About the Project

Customer churn — when a subscriber stops doing business with a company — is one of the most costly problems in the telecom industry. Acquiring a new customer can cost **5–7x more** than retaining an existing one, so identifying at-risk customers before they leave is critical for reducing revenue loss and improving retention strategies.

This project builds an **end-to-end machine learning solution** that:

1. Analyzes historical customer data to uncover the key drivers of churn.
2. Trains a robust ensemble classifier to predict which customers are likely to churn.
3. Serves predictions through a **Flask web application** where users can input customer details and get instant churn predictions.

### The Dataset

The project uses the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset — a widely used benchmark for churn modeling. It contains **7,043 customer records** with **21 features** covering:

| Category | Features |
|---|---|
| **Demographics** | Gender, Senior Citizen, Partner, Dependents |
| **Account info** | Tenure, Contract type, Payment Method, Paperless Billing, Monthly & Total Charges |
| **Services** | Phone, Multiple Lines, Internet (DSL / Fiber optic), Online Security, Online Backup, Device Protection, Tech Support, Streaming TV & Movies |
| **Target** | `Churn` — Yes / No |

The dataset exhibits a **class imbalance** (~26% churned vs ~74% retained), which the pipeline addresses with oversampling.

---

## Exploratory Data Analysis (EDA)

The EDA notebook ([notebooks/eda.ipynb](notebooks/eda.ipynb)) provides a comprehensive analysis of the dataset. Key findings:

- **Churn Rate:** ~26% of customers churned, ~74% stayed.
- **Top Churn Drivers (positive correlation):**
  - Fiber optic internet (+0.31), electronic check payments (+0.30), high monthly charges (+0.19), paperless billing (+0.19), senior citizen status (+0.15).
- **Strongest Retention Signals (negative correlation):**
  - Online security (−0.17), tech support (−0.16), having dependents (−0.16), having a partner (−0.15).
- **Notable Churn Rates by Segment:**
  - Fiber optic users: **~42%** churn rate (highest among internet types).
  - Senior citizens: **~42%** vs ~24% for non-seniors.
  - No online security / tech support: **~42%**.
  - Customers with partner or dependents: **~15–20%**.

All EDA plots and correlation files are saved in `reports/figures/` and `reports/`.

---

## Model Evaluation

Eight models were trained and evaluated in [Model_evaluation/model_evaluation.py](Model_evaluation/model_evaluation.py) to find the best approach for this churn problem. All models were trained with class-imbalance handling — either via `RandomOverSampler` or built-in `class_weight='balanced'`.

| # | Model | Accuracy | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| 1 | Logistic Regression (Balanced) | 0.7388 | **0.7834** | 0.6143 | **0.8417** |
| 2 | SVM (Balanced) | 0.7466 | 0.7807 | 0.6206 | 0.8250 |
| 3 | AdaBoost | 0.7374 | 0.7727 | 0.6097 | 0.8333 |
| 4 | Gradient Boosting | 0.7502 | 0.7620 | 0.6182 | 0.8410 |
| 5 | HistGradientBoosting | 0.7615 | 0.7193 | 0.6156 | 0.8265 |
| 6 | KNN | 0.7126 | 0.7059 | 0.5659 | 0.7680 |
| 7 | XGBoost | 0.7580 | 0.6791 | 0.5984 | 0.8045 |
| 8 | Random Forest (Balanced) | **0.7949** | 0.5000 | 0.5641 | 0.8252 |

**Key takeaways from the comparison:**

- **Recall is the priority metric** — missing a churning customer (false negative) is more costly than a false alarm, so models were ranked primarily by recall and ROC-AUC.
- **Logistic Regression** achieved the highest recall (0.78) and ROC-AUC (0.84), proving that a simpler model can outperform complex ones when properly tuned.
- **SVM and AdaBoost** followed closely in recall, while **Gradient Boosting** matched the top ROC-AUC.
- **Random Forest** had the best accuracy (0.79) but the worst recall (0.50) — it correctly classified the majority class well but missed half the churners, making it unsuitable for this use case.
- **KNN** underperformed across all metrics, suggesting the feature space doesn't favor distance-based similarity.

Based on these results, the three most complementary models — **Gradient Boosting**, **Logistic Regression**, and **SVM** — were combined into a soft voting ensemble for the final pipeline.

---

## Training Pipeline

The training flow in `train.py` uses:

- Centralized cleaning via `utils/preprocessing.py`
- `ColumnTransformer` for numeric scaling + categorical one-hot encoding
- `RandomOverSampler` to handle class imbalance
- **Soft Voting Ensemble** of three diverse classifiers:
  - `GradientBoostingClassifier` (tree-based)
  - `LogisticRegression` (linear, balanced)
  - `SVC` (kernel-based, balanced)
- Stratified 5-fold cross-validation + holdout test evaluation

### Why RandomOverSampler Instead of SMOTE?

Applying regular SMOTE after one-hot encoding can generate invalid synthetic samples with fractional dummy values. `RandomOverSampler` duplicates minority samples instead, preserving valid category combinations.

### Why a Soft Voting Ensemble?

After benchmarking 8 models in `Model_evaluation/model_evaluation.py`, the top three complementary performers were combined:

| Model | Strength |
|---|---|
| Gradient Boosting | Best ROC-AUC (0.841), strong recall |
| Logistic Regression | Best recall (0.783), different algorithm family |
| SVM | Strong recall (0.781), kernel-based diversity |

Soft voting averages predicted probabilities, letting each model compensate for the others' weaknesses.

### Data Cleaning

`clean_data(df)` in `utils/preprocessing.py` handles:

- Converting `TotalCharges` to numeric and imputing missing values with the median
- Dropping `customerID`
- Encoding `Churn` as `Yes → 1`, `No → 0`
- All operations are non-destructive (`df.copy()`)

---

## Results

From the most recent run of `train.py`:

### 5-Fold Stratified Cross-Validation

| Metric | Mean ± Std |
|---|---|
| Accuracy | 0.7551 ± 0.0085 |
| Precision | 0.5262 ± 0.0112 |
| Recall | 0.7742 ± 0.0189 |
| F1 | 0.6265 ± 0.0129 |
| ROC-AUC | 0.8470 ± 0.0130 |

### Holdout Test Set

| Metric | Score |
|---|---|
| Accuracy | 0.7523 |
| Precision | 0.5220 |
| Recall | 0.7914 |
| F1-Score | 0.6291 |
| ROC-AUC | 0.8448 |

---

## Project Structure

```
├── train.py                  # Training pipeline (preprocessing + CV + evaluation)
├── app.py                    # Flask web app for inference
├── utils/preprocessing.py    # Reusable data cleaning logic
├── Model_evaluation/         # Broader model-comparison experiments
├── notebooks/eda.ipynb       # Exploratory data analysis
├── templates/index.html      # Frontend form for predictions
├── reports/                  # EDA outputs, correlation files, figures
├── data/                     # Raw dataset
├── deploy.sh                 # One-command EC2 deployment script
├── requirements.txt
└── Dockerfile
```

---

## Getting Started

```bash
git clone https://github.com/<your-username>/customer-churn-ml.git
cd customer-churn-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Deploy on AWS EC2 (Ubuntu + Nginx)

A single script handles the full setup — system packages, Python venv, model training, gunicorn service, and nginx reverse proxy.

**Prerequisites:**
- An Ubuntu EC2 instance (e.g. `t2.micro` or `t3.small`)
- SSH access and port **80** open in the Security Group

**Steps:**

```bash
# 1. SSH into your EC2 instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# 2. Clone the repo
git clone https://github.com/<your-username>/customer-churn-ml.git
cd customer-churn-ml

# 3. Run the deploy script
chmod +x deploy.sh
sudo ./deploy.sh
```

The app will be live at `http://<ec2-public-ip>`. No manual configuration needed.

**What the script does:**
1. Installs Python 3, pip, venv, and nginx
2. Creates a virtual environment and installs all dependencies
3. Trains the model (generates `model.pkl`)
4. Sets up gunicorn as a systemd service (auto-restarts on failure)
5. Configures nginx as a reverse proxy (port 80 → gunicorn on 5000)

**Useful commands after deployment:**
```bash
# Check app status
sudo systemctl status churn-ml

# View app logs
journalctl -u churn-ml -f

# Restart after code changes
cd ~/customer-churn-ml && git pull
sudo systemctl restart churn-ml
```

---

## Notes

- In the Flask app, numeric fields are validated before prediction — invalid inputs return a user-friendly error instead of crashing.
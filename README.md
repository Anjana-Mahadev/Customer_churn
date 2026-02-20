# Customer Churn Prediction

## Project Overview
This project predicts whether a customer is likely to churn (i.e., stop using the service) for a telecom company.  
It uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and implements an end-to-end ML pipeline with a Flask web app for real-time predictions.

The workflow includes:

1. **Data Preprocessing** – handling categorical and numeric features.  
2. **Model Training** – using XGBoost Classifier for structured tabular data.  
3. **Evaluation** – calculating multiple performance metrics.  
4. **Deployment** – Flask app for user-friendly interaction.

---

## Features Used

- **Numeric:** `tenure`, `MonthlyCharges`, `TotalCharges`  
- **Categorical:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

> All features are considered relevant, as they can influence customer churn individually or in combination.

---

## Model Choice

- **XGBoost Classifier**  
- Chosen for:  
  - Handling both numeric and categorical features efficiently  
  - Robust performance on structured tabular data  
  - Built-in feature importance metrics for interpretability

---

## Performance Metrics

After training and evaluating the model on a holdout test set:

| Metric        | Value  |
|---------------|--------|
| ROC-AUC       | 0.8238 |
| Accuracy      | 0.78   |
| Precision     | 0.60   |
| Recall        | 0.53   |
| F1-Score      | 0.56   |

The ROC-AUC indicates good ranking ability, while precision and recall reflect moderate detection of churners. These metrics are suitable for a baseline production-ready model.

---

## How to Run the App
'''bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
'''
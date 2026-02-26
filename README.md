# 📊 Customer Churn Prediction System

---

## 🚀 Project Overview

This project builds an end-to-end **Customer Churn Prediction** system using Machine Learning.

The objective is to predict whether a customer is likely to churn so that the business can proactively take retention actions.

### The system includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Handling class imbalance using SMOTE
- Training multiple classification models
- Model evaluation using business-driven metrics
- Final model selection based on comparative performance

---

## 🎯 Business Objective

Customer churn prediction is a **binary classification problem**:

- `1` → Customer will churn  
- `0` → Customer will not churn  

Since churn datasets are typically imbalanced, accuracy alone is not reliable.

The main goal is to:

> **Maximize Recall (detect churners)** while maintaining strong overall discrimination ability (ROC-AUC).

---

# 🔎 Project Workflow

---

## 📊 1. Exploratory Data Analysis (EDA)

### 🔹 Churn Distribution

- The dataset is moderately imbalanced.
- Majority class: Non-churn customers.
- Minority class: Churn customers (~26–30%).
- This imbalance justified the use of SMOTE during model training.

---

### 🔹 Key Numeric Feature Insights

- **Tenure**
  - Customers with lower tenure show higher churn rates.
  - Long-term customers are less likely to churn.

- **MonthlyCharges**
  - Higher monthly charges are associated with higher churn probability.

- **TotalCharges**
  - Lower total charges correlate with higher churn (often linked to low tenure).

---

### 🔹 Categorical Feature Insights

#### High churn observed in:
- Month-to-month contracts
- Customers without Tech Support
- Customers without Online Security
- Electronic check payment method users

#### Low churn observed in:
- Two-year contracts
- Customers with additional service protections
- Long-term subscribers

---

### 🔹 Correlation Analysis

- Tenure has strong negative correlation with churn.
- MonthlyCharges has moderate positive correlation with churn.
- Contract type features show strong influence.
- Most individual features have moderate correlation, indicating churn is influenced by multiple interacting factors.

Full correlation matrix and heatmaps were generated and saved in the `reports/` folder.

---


## ⚙️ 2. Data Preprocessing

- Handled missing values
- Converted `TotalCharges` to numeric
- Encoded categorical variables
- Scaled numerical features
- Used `ColumnTransformer` for structured preprocessing
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** inside an `ImbPipeline` to handle class imbalance and prevent data leakage

---

## 🤖 3. Models Trained

To ensure robust model selection, the following models were trained and evaluated:

- Logistic Regression (Balanced)
- SVM (Balanced)
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- KNN
- XGBoost
- Random Forest (Balanced)

All models were trained using the same preprocessing pipeline to ensure fair comparison.

---

# 📈 Model Evaluation Strategy

Since this is an imbalanced classification problem, the following metrics were prioritized:

- **Recall** → Detect maximum churners (primary business metric)
- **ROC-AUC** → Overall model ranking ability
- **F1-Score** → Balance between Precision and Recall
- **Accuracy** → For reference only

---

# 📊 Final Model Comparison Results

| Model                          | Accuracy | Recall | F1-Score | ROC-AUC |
|--------------------------------|----------|--------|----------|---------|
| Logistic Regression (Balanced) | 0.7388   | 0.7834 | 0.6143   | 0.8417  |
| SVM (Balanced)                 | 0.7466   | 0.7807 | 0.6206   | 0.8250  |
| AdaBoost                       | 0.7374   | 0.7727 | 0.6097   | 0.8333  |
| Gradient Boosting              | 0.7502   | 0.7620 | 0.6182   | 0.8410  |
| HistGradientBoosting           | 0.7615   | 0.7193 | 0.6156   | 0.8265  |
| KNN                            | 0.7126   | 0.7059 | 0.5659   | 0.7680  |
| XGBoost                        | 0.7580   | 0.6791 | 0.5984   | 0.8045  |
| Random Forest (Balanced)       | 0.7949   | 0.5000 | 0.5641   | 0.8252  |

---

# 🏆 Final Model Selection

Although **Logistic Regression** achieved the highest Recall (0.7834),  
**Gradient Boosting** was selected as the final model due to its strong overall balance across metrics.

### Why Gradient Boosting?

- High ROC-AUC (0.8410)
- Strong Recall (0.7620)
- Stable F1-score
- Better generalization compared to linear models
- More robust performance trade-off

---

## ✅ Final Selected Model: Gradient Boosting

### 📌 Final Performance Metrics

- **Accuracy:** 0.7502  
- **Recall:** 0.7620  
- **F1-Score:** 0.6182  
- **ROC-AUC:** 0.8410  

---

# 💼 Business Interpretation

- The model detects ~76% of churners.
- It has strong separation ability (ROC-AUC ≈ 0.84).
- Maintains a balanced trade-off between false positives and false negatives.
- Suitable for real-world churn retention campaigns.

---

# 🛠 Tech Stack

- Python
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Pandas / NumPy
- Matplotlib / Seaborn

---

# 📌 Conclusion

This project demonstrates a complete machine learning lifecycle:

✔ Data preprocessing  
✔ Class imbalance handling  
✔ Multiple model comparison  
✔ Business-driven metric selection  
✔ Production-ready final model  

The selected **Gradient Boosting model** provides strong predictive power and is suitable for proactive churn management strategies.

# How to Run the App
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
```
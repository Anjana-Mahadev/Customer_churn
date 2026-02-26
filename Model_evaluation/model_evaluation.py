import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

# Local imports for XGBoost and SMOTE
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# 1. Load Data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify numerical and categorical columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in df.columns if col not in num_cols + ['Churn']]

# One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Define X and y
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. TARGETED SCALING (Only scale numeric features)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# 5. HANDLING IMBALANCE (SMOTE)
if SMOTE_AVAILABLE:
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
else:
    print("SMOTE not found. Using Random Over-Sampling...")
    train_data = pd.concat([X_train_scaled, y_train], axis=1)
    churn_no = train_data[train_data.Churn == 0]
    churn_yes = train_data[train_data.Churn == 1]
    churn_yes_upsampled = resample(churn_yes, replace=True, n_samples=len(churn_no), random_state=42)
    X_train_res = pd.concat([churn_no, churn_yes_upsampled]).drop('Churn', axis=1)
    y_train_res = pd.concat([churn_no, churn_yes_upsampled])['Churn']

# 6. Define Models
models = {
    "Logistic Regression (Balanced)": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest (Balanced)": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "SVM (Balanced)": SVC(probability=True, random_state=42, class_weight='balanced'),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# 7. Evaluate
results_list = []
for name, model in models.items():
    # Use internal balancing if supported, otherwise use the resampled data
    if "Balanced" in name:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train_res, y_train_res)
        
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    results_list.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

# 8. Store and Display
results_df = pd.DataFrame(results_list).sort_values(by='Recall', ascending=False)
results_df.to_csv('final_advanced_churn_results.csv', index=False)
print(results_df)
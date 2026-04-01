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
from utils.preprocessing import clean_data

# Local imports for XGBoost and SMOTE
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import RandomOverSampler
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# 1. Load Data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Preprocessing
df = clean_data(df)

# Identify numerical and categorical columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in df.columns if col not in num_cols + ['Churn']]

# Define X and y from raw data, then split before encoding to avoid schema leakage.
X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# One-Hot Encoding separately, then align test columns to train schema.
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 4. TARGETED SCALING (Only scale numeric features)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
present_num_cols = [col for col in num_cols if col in X_train.columns]
X_train_scaled[present_num_cols] = scaler.fit_transform(X_train[present_num_cols])
X_test_scaled[present_num_cols] = scaler.transform(X_test[present_num_cols])

# 5. HANDLING IMBALANCE
if ROS_AVAILABLE:
    print("Applying RandomOverSampler...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)
else:
    print("imblearn not found. Using manual random over-sampling...")
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
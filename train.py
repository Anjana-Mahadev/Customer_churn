import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from utils.preprocessing import clean_data

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = clean_data(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Column Identification
# -------------------------------
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

# -------------------------------
# Preprocessing
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        (
            "cat",
            OneHotEncoder(
                drop="first",                # match pd.get_dummies(drop_first=True)
                handle_unknown="ignore",
                sparse_output=False
            ),
            cat_cols
        ),
    ]
)

# -------------------------------
# Pipeline with safe oversampling
# -------------------------------
ensemble = VotingClassifier(
    estimators=[
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("svm", SVC(probability=True, class_weight="balanced", random_state=42)),
    ],
    voting="soft",
)

pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        # Duplicate minority rows instead of synthesizing invalid one-hot mixes.
        ("oversample", RandomOverSampler(random_state=42)),
        ("model", ensemble)
    ]
)

# -------------------------------
# Stratified Cross-Validation
# -------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False,
)

print("\n--- 5-Fold Stratified CV ---")
for metric in scoring:
    scores = cv_results[f"test_{metric}"]
    print(f"{metric.upper():>9}: {scores.mean():.4f} +/- {scores.std():.4f}")

# -------------------------------
# Train
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n--- Soft Voting Ensemble (GB + LR + SVM) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(pipeline, "model.pkl")
print("\n✅ Model saved as model.pkl")
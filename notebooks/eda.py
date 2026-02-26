# ==========================================
# 📊 EDA + CORRELATION SCRIPT (FINAL CLEAN)
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1️⃣ Setup folders
# ------------------------------------------

os.makedirs("../reports/figures", exist_ok=True)

# ------------------------------------------
# 2️⃣ Load Dataset
# ------------------------------------------

DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("✅ Dataset Loaded Successfully")
print("Original Shape:", df.shape)

# ------------------------------------------
# 3️⃣ Drop customerID (Prevent Leakage)
# ------------------------------------------

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

print("Columns after drop:", df.columns)

# ------------------------------------------
# 4️⃣ Convert Target to Numeric
# ------------------------------------------

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("✅ Churn converted to numeric")

# ------------------------------------------
# 5️⃣ Fix TotalCharges (Convert to numeric)
# ------------------------------------------

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# ------------------------------------------
# 6️⃣ Handle Missing Values (Safe Method)
# ------------------------------------------

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing values handled safely")

# ------------------------------------------
# 7️⃣ Target Distribution (Imbalance Check)
# ------------------------------------------

plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("../reports/figures/churn_distribution.png", bbox_inches="tight")
plt.close()

print("\nChurn Distribution (%):")
print(df["Churn"].value_counts(normalize=True) * 100)

# ------------------------------------------
# 8️⃣ Numerical Feature Distributions
# ------------------------------------------

for col in numeric_cols:
    if col != "Churn":
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"../reports/figures/{col}_distribution.png", bbox_inches="tight")
        plt.close()

print("✅ Numerical feature plots saved")

# ------------------------------------------
# 9️⃣ Categorical Feature Distributions
# ------------------------------------------

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.xticks(rotation=45)
    plt.title(f"{col} Count")
    plt.savefig(f"../reports/figures/{col}_count.png", bbox_inches="tight")
    plt.close()

print("✅ Categorical feature plots saved")

# ------------------------------------------
# 🔟 One-Hot Encoding (For Correlation)
# ------------------------------------------

df_corr = pd.get_dummies(df, drop_first=True)

print("Total columns after encoding:", df_corr.shape[1])

# Safety check
if any("customerID" in col for col in df_corr.columns):
    print("❌ ERROR: customerID still present!")
else:
    print("✅ No customerID columns present")

# ------------------------------------------
# 1️⃣1️⃣ Correlation Matrix
# ------------------------------------------

corr_matrix = df_corr.corr()
corr_matrix.to_csv("../reports/full_correlation_matrix.csv")

print("✅ Full correlation matrix saved")

# ------------------------------------------
# 1️⃣2️⃣ Correlation with Target
# ------------------------------------------

target_corr = corr_matrix["Churn"].sort_values(ascending=False)

target_corr.to_csv("../reports/churn_feature_correlation.csv")

plt.figure(figsize=(8, 14))
sns.heatmap(target_corr.to_frame(), annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlation with Churn")
plt.savefig("../reports/figures/correlation_heatmap.png", bbox_inches="tight")
plt.close()

print("✅ Target correlation heatmap saved")

print("\n🔝 Top 10 Positively Correlated Features:")
print(target_corr.head(10))

print("\n🔻 Top 10 Negatively Correlated Features:")
print(target_corr.tail(10))

# ------------------------------------------
# 1️⃣3️⃣ Final Summary
# ------------------------------------------

print("\n📌 EDA SUMMARY")
print("- Dataset loaded successfully.")
print("- customerID removed to prevent leakage.")
print("- Churn converted to numeric.")
print("- Missing values handled safely.")
print("- Class imbalance checked.")
print("- Distributions plotted.")
print("- Correlation analysis completed.")
print("- Key churn-driving features identified.")

print("\n✅ All EDA outputs saved in ../reports/")
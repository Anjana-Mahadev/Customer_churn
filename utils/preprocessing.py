import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans Telco churn dataset
    """
    # Work on a copy to avoid mutating caller-owned DataFrames.
    df = df.copy()

    # Convert TotalCharges to numeric and impute only this known problematic field.
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Drop customerID (not useful for ML)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Encode target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

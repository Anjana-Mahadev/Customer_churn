import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans Telco churn dataset
    """
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop missing values
    df.dropna(inplace=True)

    # Drop customerID (not useful for ML)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Encode target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

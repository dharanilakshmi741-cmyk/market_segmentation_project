import pandas as pd

def create_rfm_features(df):
    """
    Simulated RFM Features
    Since actual purchase history is unavailable,
    we simulate Frequency & Monetary values.
    """

    rfm = pd.DataFrame()

    rfm['CustomerID'] = df['CustomerID']

    # Recency = Age (simulated)
    rfm['Recency'] = df['Age']

    # Frequency = Spending Score
    rfm['Frequency'] = df['Spending Score (1-100)']

    # Monetary = Annual Income
    rfm['Monetary'] = df['Annual Income (k$)']

    return rfm
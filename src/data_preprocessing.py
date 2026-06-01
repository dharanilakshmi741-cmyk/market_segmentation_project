import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Encode Gender column
    encoder = LabelEncoder()
    df['Gender'] = encoder.fit_transform(df['Gender'])

    # Features for clustering
    features = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    # Standardize data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return df, scaled_features
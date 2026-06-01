import os
import pandas as pd

# Create outputs folder automatically
os.makedirs('outputs', exist_ok=True)

from src.data_preprocessing import load_and_preprocess_data
from src.rfm_analysis import create_rfm_features
from src.pca_analysis import apply_pca
from src.clustering import find_optimal_clusters, perform_clustering
from src.visualization import (
    plot_elbow_method,
    plot_customer_clusters,
    plot_pca_clusters
)

def main():

    print("\n===== MARKET CUSTOMER SEGMENTATION PROJECT =====\n")

    # Load and preprocess data
    df, scaled_data = load_and_preprocess_data(
        'dataset/Mall_customers.csv'
    )

    # RFM Analysis
    rfm = create_rfm_features(df)

    # Find optimal clusters
    inertia = find_optimal_clusters(scaled_data)

    # Plot elbow method
    plot_elbow_method(inertia)

    # Perform clustering
    clusters = perform_clustering(scaled_data)

    # Add cluster labels
    df['Cluster'] = clusters

    # PCA
    pca_data = apply_pca(scaled_data)

    # Create visualizations
    plot_customer_clusters(df)

    plot_pca_clusters(pca_data, clusters)

    # Save report
    df.to_csv('outputs/cluster_report.csv', index=False)

    # Print customer data with clusters
    print("===== CUSTOMER DATA WITH CLUSTERS =====\n")
    print(df)

    # Cluster Summary
    cluster_summary = df.groupby('Cluster')[
        ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    ].mean()

    print("\n===== CLUSTER SUMMARY =====\n")
    print(cluster_summary)

    print("\nProject Completed Successfully")
    print("Outputs saved inside outputs/ folder")

if __name__ == "__main__":
    main()

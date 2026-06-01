import matplotlib.pyplot as plt

def plot_elbow_method(inertia):
    plt.figure(figsize=(8, 5))

    plt.plot(range(1, 11), inertia, marker='o')

    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')

    plt.savefig('outputs/elbow_method.png')

def plot_customer_clusters(df):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster'],
        cmap='viridis'
    )

    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')

    plt.title('Customer Segments')

    plt.savefig('outputs/customer_clusters.png')

def plot_pca_clusters(pca_data, clusters):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=clusters,
        cmap='rainbow'
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    plt.title('PCA Cluster Visualization')

    plt.savefig('outputs/pca_clusters.png')
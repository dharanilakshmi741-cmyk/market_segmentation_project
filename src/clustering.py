from sklearn.cluster import KMeans

def find_optimal_clusters(data):
    inertia = []

    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(data)
        inertia.append(model.inertia_)

    return inertia

def perform_clustering(data, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    clusters = model.fit_predict(data)

    return clusters
from sklearn.decomposition import PCA

def apply_pca(data):
    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(data)

    return principal_components
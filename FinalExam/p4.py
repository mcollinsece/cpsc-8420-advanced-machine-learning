import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

def generate_concentric_data(n_points=150, noise=0.25):
    radii = [1.0, 2.8, 5.0]
    data = []
    labels = []
    
    for i, r in enumerate(radii):
        angles = np.random.uniform(0, 2*np.pi, n_points)
        x = r * np.cos(angles) + np.random.normal(0, noise, n_points)
        y = r * np.sin(angles) + np.random.normal(0, noise, n_points)
        data.extend(list(zip(x, y)))
        labels.extend([i] * n_points)
    
    return np.array(data), np.array(labels)

def compute_knn_similarity(X, k=10):
    A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
    A = A.toarray()
    W = np.maximum(A, A.T)
    return W

def spectral_clustering(X, k_neighbors=10, n_clusters=3):
    W = compute_knn_similarity(X, k=k_neighbors)
    
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    
    eigenvals, eigenvecs = np.linalg.eigh(L)
    idx = np.argsort(eigenvals)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    Y = eigenvecs[:, 1:3]
    Y = Y / np.sqrt(np.sum(Y**2, axis=1))[:, np.newaxis]
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    labels = kmeans.fit_predict(Y)
    
    centers = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        mask = labels == i
        centers[i] = np.mean(X[mask], axis=0)
    radii = np.sqrt(np.sum(centers**2, axis=1))
    order = np.argsort(radii)
    new_labels = np.zeros_like(labels)
    for i, old_label in enumerate(order):
        new_labels[labels == old_label] = i
    
    return new_labels, eigenvals, eigenvecs

np.random.seed(42)
X, true_labels = generate_concentric_data()
labels, eigenvals, eigenvecs = spectral_clustering(X)

colors = ['orange', 'skyblue', 'lightgreen']
custom_cmap = ListedColormap(colors)

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

axs[0,0].scatter(X[:,0], X[:,1], c=labels, cmap=custom_cmap)
axs[0,0].set_title('Data and Clustering')
axs[0,0].set_xlim(-6, 6)
axs[0,0].set_ylim(-6, 6)

axs[0,1].plot(range(1, 16), eigenvals[1:16], 'go-')
axs[0,1].set_title('Eigenvalues')

n_points = len(X)
axs[1,0].set_title('Eigenvectors')
axs[1,0].set_xlabel('Index')
axs[1,0].set_ylabel('2nd/3rd Smallest')

axs[1,0].scatter(range(n_points), eigenvecs[:, 1], c=labels, cmap=custom_cmap, s=10, alpha=0.6)
axs[1,0].scatter(range(n_points), eigenvecs[:, 2], c=labels, cmap=custom_cmap, s=10, alpha=0.6)
axs[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axs[1,0].set_ylim(-0.1, 0.1)
axs[1,0].set_xlim(0, 450)

axs[1,1].scatter(eigenvecs[:, 1], eigenvecs[:, 2], c=labels, cmap=custom_cmap)
axs[1,1].set_title('Spectral Clustering')

plt.tight_layout()
plt.show() 
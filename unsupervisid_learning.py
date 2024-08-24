
################################ K-Means ###############################

import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, n_iterations=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.tol = tol  # to'liq konvergensiya uchun minimal o'zgarish
    
    def fit(self, X):
        # Raqamli markazlarni tasodifiy tanlash
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.n_iterations):
            # Har bir namunani eng yaqin markazga ajratish
            self.labels = np.argmin(self._compute_distances(X, self.centroids), axis=1)
            
            # Har bir klaster markazini yangilash
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # Konvergensiya tekshiruvi
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            
            self.centroids = new_centroids

    def _compute_distances(self, X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def predict(self, X):
        return np.argmin(self._compute_distances(X, self.centroids), axis=1)
    
# Ma'lumotlarni yaratish
np.random.seed(42)
X1 = np.random.randn(100, 2) + np.array([2, 2])
X2 = np.random.randn(100, 2) + np.array([-2, -2])
X3 = np.random.randn(100, 2) + np.array([2, -2])
X = np.vstack([X1, X2, X3])

# Model yaratish va o'qitish
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Bashoratlar qilish
y_pred = kmeans.predict(X)

# Natijalarni chizish
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', edgecolors='k')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, c='red', marker='X')
plt.title('K-Means Clustering (Scratch)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

import sklearn
from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris['data']
y = iris['target']

# Налаштування класу KMeans
KMeans(n_clusters=8, init='k-means++', n_init=0, max_iter=300,
       tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')

# Створення об'єкту KMeans та визначчення кількості кластерів
kmeans = KMeans(n_clusters=3)

# Навчання моделі кластеризації KMeans
kmeans.fit(X)


# Передбачення вихідних міток для всіх точок
y_kmeans = kmeans.predict(X)

# Відображення центрів кластерів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# Знаходженння кластерів
def find_clusters(X, n_clusters, rseed=2):
    # Випадково обираємо кластери
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # Призначаємо мітки на основі найближчого центру
        labels = pairwise_distances_argmin(X, centers)
        # Знаходимо нові центри за допомогою точок
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        # Перевіряємо центри на збіжність
        if np.all(centers == new_centers):
            break
        centers = new_centers
        return centers, labels


# Знаходження центрів кластерів та зображення їх на графіку
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Передбачення вхідних точок кластерів та зображення їх на графіку
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

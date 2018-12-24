
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

# Data Creation
X, y = make_blobs(n_samples=100,n_features=2,centers=3, cluster_std=0.5,shuffle=True, random_state=0)

# Plot the graph
#plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
#plt.show()

# plot for KMeans
n = 11
arr_X = []
arr_Y = []
for i in range(2, n):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X, y)
    arr_Y.append(kmeans.inertia_)
    arr_X.append(i)

# Plot for Inertia from K=2 to 10
#plt.scatter(arr_X, arr_Y, marker='o', edgecolor='k')
#plt.show()

# The best value for k was obtained for k=10 because k increases the distance of the points from the centroids keep on decreasing
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
y_means = kmeans.predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=y_means, cmap='viridis')
#plt.show()

# Silhouette Score
n = 11
arr_X = []
arr_Y = []
for i in range(2, n):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X, y)
    y_pred = kmeans.predict(X)
    arr_Y.append(silhouette_score(X, y_pred))
    arr_X.append(i)

# Plot for Silhouette Coefficient from K=2 to 10
plt.scatter(arr_X, arr_Y, marker='o', edgecolor='k')
plt.show()



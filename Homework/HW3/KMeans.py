import random as rand
import math
import matplotlib.pyplot as plt
import numpy as np

class kMeans:

    def __init__(self, k=1, random_state=1):
        self.K = k
        self.random_state = random_state
        rand.seed(random_state)

    def getClusters(self, X, c):
        C = []
        for i in range(len(c)):
            C.append([])

        # Assigning the points to the nearest centroids
        for i in range(len(X)):
            closest = -1
            closestDist = 10000

            # Looping over all the centroids
            j = 0
            for j in range(0, K):
                distance = math.sqrt((X[i][0] - c[j][0]) ** 2 + (X[i][1] - c[j][1]) ** 2)
                if distance < closestDist:
                    closestDist = distance
                    closest = j
            C[closest].append(X[i])
        return C

    def fit(self, X):
        C = []
        c = []

        picked = [False] * len(X)

        # Pick k random centroids
        for i in range(self.K):
            while True:
                index = rand.randint(0, len(X) - 1)
                if picked[index]: continue
                c.append([X[index][0], X[index][1]])
                picked[index] = True
                break

        print("Initial Centroids --> ")
        print(c)

        # Initial Centroids
        prevClusters = []

        # Stopping condition: if the centroids do not move
        if c != prevClusters:
            C = self.getClusters(X, c)

            # Assign the old centroids
            prevClusters = c

            # Getting new centroids
            c = []
            for i in range(len(C)):
                lst = C[i]
                dimension1 = 0
                dimension2 = 0
                for j in range(len(lst)):
                    dimension1 += lst[j][0]
                    dimension2 += lst[j][1]
                dimension1 /= len(lst)
                dimension2 /= len(lst)
                c.append([dimension1, dimension2])


        # For Plotting the graph
        color = ['red', 'blue', 'orange', 'purple', 'black', 'grey', 'green']
        for i in range(len(C)):
            x = []
            y = []
            lst = C[i]
            for j in range(len(lst)):
                x.append(lst[j][0])
                y.append(lst[j][1])
                plt.scatter(x, y, color=color[i])

        plt.show()

oldFaithfulData = [
    [3.6, 79],
    [1.8, 54],
    [2.283, 62],
    [3.333, 74],
    [2.883, 55],
    [4.533, 85],
    [1.950, 51],
    [1.833, 54],
    [4.7, 88],
    [3.6, 85],
    [1.600, 52],
    [4.350, 85],
    [3.917, 84],
    [4.2, 78],
    [1.750, 62],
    [1.8, 51],
    [4.7, 83],
    [2.167, 52],
    [4.800, 84],
    [1.750, 47],
]

X = np.array([[2,4],[2,6], [2,8], [10,4], [10,6], [10,8]])
K = 2

# When the cluster centroids are selected on the same side
#kM = kMeans(K, 15)
#kM.fit(X)

# When the cluster centroids are selected on the different side
kM = kMeans(K, 1)
kM.fit(X)






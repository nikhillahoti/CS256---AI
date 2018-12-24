# step 1

# step 2
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, cluster_std=2.0, centers=2)
print(X.shape)
print(y.shape)

# step 3: Scatter plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
X1, X2 = X.T

color1 = 0.4
color2 = 0.8
colors = [color1 if y[i] == 0 else color2 for i in range(len(y))]
plt.scatter(X1, X2, c=colors)
plt.show()

# step 4: Split the data into training and testing sets using 30% split criteria
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# step 5: Train using Gaussian NB Classifier
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

# step 6: Testing
from sklearn.metrics import accuracy_score
y_test_predict = model.predict(X_test)
print('Test Accuracy: %.2f' %accuracy_score(y_test, y_test_predict))

y_train_predict = model.predict(X_train)
print('Train Accuracy: %.2f' %accuracy_score(y_train, y_train_predict))

"""
Sample Output
/home/nikhil/PycharmProjects/CodingPrac/venv/bin/python /home/nikhil/PycharmProjects/CodingPrac/venv/GaussianNaiveBayesClassifier.py
(100, 2)
(100,)
Test Accuracy: 1.00
Train Accuracy: 1.00
"""







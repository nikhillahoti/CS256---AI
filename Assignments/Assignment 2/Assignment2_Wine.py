# Step 1
from sklearn import datasets
wine = datasets.load_wine()


# Step 2
print(wine.feature_names)


# Step 3
print(wine.target_names)


# Step 4
from sklearn.datasets import load_wine
X, Y = load_wine(return_X_y=True)


# Step 5
print(type(X))
print(type(Y))


# Step 6: Getting the dimensions
print(X.shape)
print(Y.shape)


# Step 7
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.30, random_state=1)

# random_state is used to define RandomState
# When passed None or np.random, it returns randomly initialized RandomState object
# When passed interger, the given integer value is used to seed new RandomState object
# When a RandomState is passed, the object is used as is


# Step 8
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred_test = dtree.predict(X_test)


#Step 9
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))


#Step 10
x_pred_test = dtree.predict(X_train)
print('Accuracy: %.2f' %accuracy_score(y_train, x_pred_test))


# Step 11
# For test data: it varies (because of the seed) 0.96
# For train data: it is 1
# Accuracy for Train is better has our model is perfectly fitted on that data



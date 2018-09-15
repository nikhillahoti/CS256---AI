

# Importing Packages
import pandas as pd
from sklearn.model_selection import train_test_split

# 1:
X1 = [0,0,1,1,1,1,0,0]
X2 = [0,1,0,1,0,1,1,0]
X3 = [0,0,1,0,0,1,1,1]
# 2: Creating Input DataFrame from the lists
X = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

# Creating Output Dataframe
Y = pd.DataFrame({'Y': [1,1,0,1,1,0,0,0]})


# 3: Printing the DataFrames
print(X)
print(Y)


# 4: Column Values
print(X.columns.values)
print(Y.columns.values)

# 5: Splitting the data into two sets: Train and Test
# It splits the dataset randomly in ratio of 70:30 (Train: Test). The fours lists are then stored into respective lists
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# 6: Displaying the Training datasets
print(X_train)
print(y_train)

# 7: importing Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

# 8: training the model
dtree.fit(X_train, y_train)

# 9: Predict the values
dtree.predict(X_test)

# 10: Displaying the results
features = X.columns
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree,
out_file=dot_data,
feature_names=features,
class_names=['0','1'],
filled=True, rounded=True,
impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree.png")




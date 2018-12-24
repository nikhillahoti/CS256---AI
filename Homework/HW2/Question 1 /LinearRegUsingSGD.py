import numpy as np
import time
from sklearn.model_selection import train_test_split

class linearReg:
    def __init__(self, epochs=10000, learning_rate=0.0001, random_state=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        start = time.time()
        self.learning_parameters = np.random.uniform(0, 1, X.shape[-1] + 1)
        print "Starting Parameters -->"
        print(self.learning_parameters)
        for iteration in range(self.epochs):
            sum_error = 0.0
            for i in range(len(X)):
                yhat = self.predict_Single_Row(X[i])
                error = yhat - y[i]
                sum_error += error ** 2
                self.learning_parameters[0] -= self.learning_rate * error
                for noParams in range(len(X[i])):
                    self.learning_parameters[noParams + 1] -= self.learning_rate * error * X[i][noParams]
            print("Total Error for epoch -> " + str(iteration) + " -> " + str(sum_error))
        print "Final Values "
        print(self.learning_parameters)
        end = time.time()
        print "Total Time --> ", (end - start)

    def predict(self, X):
        prediction = []
        for i in range(len(X)):
            prediction.append(self.predict_Single_Row(X[i]))
        return prediction


    def predict_Single_Row(self, x):
        value = self.learning_parameters[0]
        for i in range(1, len(self.learning_parameters)):
            value += self.learning_parameters[i] * x[i - 1]
        return int(round(value))

    def accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

X1= np.random.randint(1,10, 100)
X2= np.random.randint(1,10,100)

Y= 4*X1 + 10 *X2
X = np.hstack((X1.T,X2.T))
X1=np.vstack(X1)
X2=np.vstack(X2)
X= np.hstack((X1,X2))
Y = np.vstack(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

lR = linearReg()
lR.fit(X_train, y_train)
print lR.accuracy(lR.predict(X_test), y_test.reshape(y_test.shape[0]))

#for i in range(len(X_test)):
#    print(lR.predict_Single_Row(X_test[i]), y_test[i])


import numpy as np
import math
import operator

class DecisionStump:

    classLabels = []
    Attributes = {}
    lowestIndex = -1

    def calculateforClass(self, X):
        dict = {}
        for i in range(len(X)):
            if X[i] in dict:
                dict[X[i]] += 1
            else:
                dict[X[i]] = 1

        logValue = 0
        for key in dict:
            logValue += ((dict[key] / len(X)) * -1 * (math.log2(dict[key] / len(X))))
        return logValue

    def assignLabel(self, X, Y):
        dict, dictClassCount = self.getCountDictionaries(X, Y)

        for key in dict:
            self.Attributes[key] = max(dictClassCount[key].items(), key=operator.itemgetter(1))[0]

    def getEmptyDict(self):
        dict = {}
        for i in range(len(self.classLabels)):
            dict[self.classLabels[i]] = 0
        return dict

    def getCountDictionaries(self, X, Y):
        # Contains the number of times the feature values occured
        # for e.g. Sunny: 6, Overcast: 4, Rain: 5
        dict = {}

        # It is a dictionary of dictionary which contains the number of times the output labels classes occured for each feature value
        # for e.g. sunny: {'no': 5, 'yes': 1}
        # This is needed for the log calculcations below
        dictClassCount = {}
        for i in range(len(X)):
            if X[i] in dict:
                dict[X[i]] += 1
            else:
                dict[X[i]] = 1
                dictClassCount[X[i]] = self.getEmptyDict()

            dictClassCount[X[i]][Y[i]] += 1

        return (dict, dictClassCount)

    def calculateLog(self, X, Y):
        dict, dictClassCount = self.getCountDictionaries(X, Y)

        logValue = 0
        for key in dict:
            for label in dictClassCount[key]:
                # checking case when the log might get to 0 which is indeterminate
                if dictClassCount[key][label] == dict[key] or dictClassCount[key][label] == 0:
                    continue
                logValue += (dict[key] / len(X)) * ((float(dictClassCount[key][label]) / dict[key])* -1 * (math.log2(dictClassCount[key][label] / dict[key])))

        return logValue

    def fit(self, X, Y):

        # Checking the boundary conditions:
        if len(X) == 0 or len(Y) == 0:
            print("Incorrect Data! Training failed!")
            return

        # It first Calculates the unique labels of the output class
        self.classLabels = list(set(Y))

        # Information Gain H(S) is calculated
        HS = self.calculateforClass(Y)

        # Then the feature with most Information Gain is selected. This feature number is stored in the lowest variable
        lowest = self.calculateLog(X[:,0], Y)
        self.lowestIndex = 0
        for i in range(1, len(X[0])):
            currLowest = self.calculateLog(X[:, i], Y)
            if currLowest < lowest:
                lowest = currLowest
                self.lowestIndex = i

        # A dictionary is then created which contains the output class associated against the class
        # for e.g. in case of the Outlook  feature, this dictionary will contain the labels as follows:
        # 'sunny': 'no' as 5 out of 6 are 'no' output labels
        # similarly 'overcast': 'yes' and
        # 'rain': 'yes'
        self.assignLabel(X[:, self.lowestIndex], Y)

    def predict(self, X):
        if self.lowestIndex == -1:
            print("Model not trained yet")
            return

        selectedFeature = X[:, self.lowestIndex]
        predictions = []
        for i in range(len(selectedFeature)):
            predictions.append(self.Attributes[selectedFeature[i]])
        return predictions

X = np.array([['sunny', 'hot', 'high', 'weak'],
             ['sunny', 'hot', 'high', 'strong'],
             ['overcast', 'hot', 'high', 'weak'],
             ['rain', 'mild', 'high', 'weak'],
             ['rain', 'cool', 'normal', 'weak'],
             ['rain', 'cool', 'normal', 'strong'],
             ['overcast', 'cool', 'normal', 'strong'],
             ['sunny', 'mild', 'high', 'weak'],
             ['sunny', 'cool', 'normal', 'weak'],
             ['rain', 'mild', 'normal', 'weak'],
             ['sunny', 'mild', 'normal', 'strong'],
             ['overcast', 'mild', 'high', 'strong'],
             ['overcast', 'hot', 'normal', 'weak'],
             ['rain', 'mild', 'high', 'strong'],
             ['sunny', 'mild', 'high', 'strong']])

Y = np.array(['no',
              'no',
              'yes',
              'yes',
              'yes',
              'no',
              'yes',
              'no',
              'yes',
              'yes',
              'no',
              'yes',
              'yes',
              'no',
              'no']
             )

DecStump = DecisionStump()
DecStump.fit(X, Y)

X_Test = np.array([['sunny', 'hot', 'high', 'weak'],
             ['sunny', 'hot', 'high', 'strong'],
             ['overcast', 'hot', 'high', 'weak']])
Y_Prediction = DecStump.predict(X_Test)
print(Y_Prediction)
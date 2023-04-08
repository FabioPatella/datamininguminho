# This is a sample Python script.
import itertools
# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import math
from collections import defaultdict
import numpy as np

from collections import defaultdict
import numpy as np
import pandas as pd


class NaiveBayesClassifier:
    def __init__(self, pseudocount=0, use_log_probs=False):
        self.pseudocount = pseudocount
        self.use_log_probs = use_log_probs
        self.class_counts = None
        self.feature_counts = None
        self.feature_probs = None
        self.classes = []
        self.columnnames=None

    def fit(self, X, y,columnnames):
        self.columnnames=columnnames
        X = pd.DataFrame(X,columns=columnnames)
        self.class_counts = defaultdict(int) #dictionary that returns 0 when a non-existing key is accessed
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: np.zeros(len(set(y))) + self.pseudocount)) #for each class label the count of the feature values, if no feature value exists it returns an array with the pseudocount
        for class_label in set(y):
            subX_classlabel = X[y == class_label]  #takes all the rows
            self.classes.append(class_label)
            self.class_counts[class_label] = subX_classlabel.shape[0] #store the number of records for the class_label
            for feature_index, feature_name in enumerate(X.columns):
                for feature_value in set(X[feature_name]):
                    self.feature_counts[feature_index][feature_value][class_label] += np.sum(
                        subX_classlabel[feature_name] == feature_value)
        self.feature_probs = {}
        class_counts = np.array(list(self.class_counts.values()))
        for feature_index, feature_name in enumerate(X.columns):
            feature_value_counts = self.feature_counts[feature_index]
            self.feature_probs[feature_name] = {}
            for feature_value, feature_counts in feature_value_counts.items():


                if self.use_log_probs:
                    self.feature_probs[feature_name][feature_value] = np.log(
                        feature_counts / (class_counts + 2 * self.pseudocount))
                else:
                    self.feature_probs[feature_name][feature_value] = feature_counts / (
                                class_counts + 2 * self.pseudocount)

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.columnnames)
        predictions = []
        for _, row in X.iterrows():
            class_probs = {}
            for class_label in self.classes:
                class_prob = np.log(self.class_counts[class_label] / X.shape[0])
                for feature_name, feature_value in row.items():

                    feature_prob = self.feature_probs[feature_name][feature_value][class_label]
                    class_prob += feature_prob
                class_probs[class_label] = class_prob
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return predictions

    def __repr__(self,valueencoding):
        valueencoding = {value: key for key, value in valueencoding.items()}
        feature_probs={}
        for featurekey in self.feature_probs.keys():
            valuesdict= self.feature_probs[featurekey]
            newdict={}
            for valuekey in valuesdict.keys():
                newdict[valueencoding[valuekey]]=valuesdict[valuekey]
            feature_probs[featurekey]=newdict
        print(feature_probs)
        self.generate_decision_rules(feature_probs)



    def generate_decision_rules(self, feature_probs):
            feature_names = list(feature_probs.keys())
            probs = []
            for c in self.classes:
                for features in itertools.product(*[list(feature_probs[f].keys()) for f in feature_names]):
                    prob = 1
                    for f in feature_names:
                        if f in features:
                            prob *= feature_probs[f][features[features.index(f)]][c]
                    probs.append(prob)
                    rule = "IF "
                    for i, f in enumerate(feature_names):
                        rule += f.upper() + " = " + str(features[i]).upper()
                        if i < len(feature_names) - 1:
                            rule += " AND "
                    rule += " THEN class = " + str(c)
                    print(rule)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define the input features
    outlook = ['Overcast', 'Overcast', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Overcast', 'Rain',
               'Sunny', 'Sunny', 'Rain', 'Sunny']
    temperature = ['Hot', 'Mild', 'Mild', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Hot', 'Cool', 'Hot', 'Hot', 'Mild',
                   'Mild']
    humidity = ['Normal', 'High', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'High', 'High', 'Normal', 'High',
                'High', 'High', 'High']
    wind = ['Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak',
            'Strong', 'Weak']

    # Define the target variable
    play_tennis = ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No']

    # Convert input features to numerical data
    outlook_num = np.array([12, 12, 13, 14, 13, 12, 14, 14, 12, 14, 13, 13, 14, 13])
    temperature_num = np.array([15, 4, 4, 4, 5, 5, 5, 4, 15, 5, 15, 15, 4, 4])
    humidity_num = np.array([6, 7, 6, 6, 6, 6, 6, 7, 7, 6, 7, 7, 7, 7])
    wind_num = np.array([8, 9, 9, 8, 8, 9, 8, 8, 8, 9, 9, 8, 9, 8])

    # Combine input features into X matrix
    X = np.vstack((outlook_num, temperature_num, humidity_num, wind_num)).T
    Xnames = np.vstack((outlook, temperature, humidity, wind))
    ynames = play_tennis

    # Convert target variable to numerical data
    play_tennis_num = np.array([1 if x == 'Yes' else 0 for x in play_tennis])

    # Define target variable y
    y = play_tennis_num
    value_encoding = {
        'Sunny': 13, 'Overcast': 12, 'Rain': 14, 'Hot': 15, 'Mild': 4, 'Cool': 5, 'Normal': 6
        , 'High': 7, 'Weak': 8,
        'Strong': 9, 'No': 0, 'Yes': 1
    }
    naiveBayesClassifier = NaiveBayesClassifier()
    column_names= ['outlook','temperature','humidity','wind']
    naiveBayesClassifier.fit(X,y,column_names)
    naiveBayesClassifier.__repr__(value_encoding)
    # predicting on new data
    outlook_num = np.array([13, 14, 12, 14, 13])
    temperature_num = np.array([5, 15, 4, 5, 15])
    humidity_num = np.array([6, 7, 7, 6, 7])
    wind_num = np.array([8, 9, 9, 8, 8])
    X = np.vstack((outlook_num, temperature_num, humidity_num, wind_num)).T
    print(naiveBayesClassifier.predict(X))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

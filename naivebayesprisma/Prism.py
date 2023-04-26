from collections import Counter
import numpy as np


class Prism:
    def __init__(self, valueencoding, columnnames):
        """
            Initializes a new instance of the Prism class.
            Args:
            - valueencoding (dict): a dictionary containing the encoded values for each attribute
            - columnnames (list): a list of the names of the columns in the dataset
            """
        self.attribute_names = None
        self.rules = []
        self.valueencoding = {value: key for key, value in valueencoding.items()}
        self.columnnames = columnnames

    def fit(self, X, y):
        """
            Fits the model to the dataset.

            Args:
            - X (array): a numpy array containing the features of the dataset
            - y (array): a numpy array containing the labels of the dataset

            """
        self.X = X
        self.y = y
        self.probabilities = {}

        for class_label in set(y):
            X = self.X
            y = self.y
            while (class_label in y):
                self.computerule(X, y, class_label, [])
                lastrule = self.rules[-1][0]
                X, y = self.remove_instances(X, y, lastrule)

        print(self.rules)

    def remove_instances(self, X, y, rule):
        """
           Removes instances from the dataset that do not satisfy the rule.

           Args:
           - X (array): a numpy array containing the features of the dataset
           - y (array): a numpy array containing the labels of the dataset
           - rule (list): a list of tuples representing the rule

           Returns:
           - newX (array): a numpy array containing the features of the filtered dataset
           - newy (array): a numpy array containing the labels of the filtered dataset
        """
        xrows = []
        newy = []
        for indexrow in range(X.shape[0]):
            toappend = False
            for featurevalue in rule:

                featureindex = featurevalue[0]
                value = featurevalue[1]
                if X[indexrow, featureindex] != value: toappend = True
            if (toappend):
                xrows.append(X[indexrow])
                newy.append(y[indexrow])
        newX = np.vstack(xrows)
        newy = np.array(newy)
        return newX, newy

    def computerule(self, X, y, class_label, featurevaluelist):
        """
         Computes a rule for the specified class label.
        :param X:a numpy array containing the features of the dataset
        :param y: a numpy array containing the output label of the dataset
        :param class_label: the class label for the speficic computation
        :param featurevaluelist: a list of current feature value mapping
        """
        stop = all(element == class_label for element in y)
        if (stop):
            self.rules.append((featurevaluelist, class_label))
            return
        for attribute_index in range(X.shape[1]):
            submatrixlabattr = X[:, attribute_index]
            values = np.unique(submatrixlabattr)
            valueprobability = {}
            for value in values:
                total_count = np.count_nonzero(submatrixlabattr == value)
                indicesoflabelrows = np.where((submatrixlabattr == value) & (y == class_label))[0]
                partial_count = len(indicesoflabelrows)
                attributevalueprobability = partial_count / total_count
                valueprobability[value] = attributevalueprobability
            self.probabilities[attribute_index] = valueprobability
        feature, value = self.find_max_prob(self.probabilities)
        newX = X[X[:, feature] == value, :]
        newY = y[X[:, feature] == value]
        featurevaluelist.append((feature, value))
        self.computerule(newX, newY, class_label, featurevaluelist)

    def find_max_prob(self, prob_dict):
        """
        Function to find the feature with the maximum probability
        :param prob_dict: dictionary conataining probabilities
        :return: a pair feature value with maximum probability
        """
        max_prob = 0.0
        max_feature = None
        max_value = None

        for feature, values in prob_dict.items():
            for value, probability in values.items():
                if probability > max_prob:
                    max_prob = probability
                    max_feature = feature
                    max_value = value

        return max_feature, max_value

    def convertprobabilities(self):
        """
        Function to convert probabilities to the original values of the features and print them
        """
        converted_dict = {}
        index = 0
        for keyfeature in self.probabilities:

            value_dict = self.probabilities[keyfeature]
            new_dict = {}
            for keyvalue in value_dict:
                new_dict[self.valueencoding[keyvalue]] = value_dict[keyvalue]
            converted_dict[self.columnnames[index]] = new_dict
            index = index + 1
        print(converted_dict)

    def predict(self, sample):
        """
        Function to predict the class label of a sample using the rules generated from fit() function
        """
        for rule in self.rules:
            applicable = True
            for featurevalue in rule[0]:
                featureindex = featurevalue[0]
                value = featurevalue[1]
                if (sample[featureindex]) != value: applicable = False
            if applicable: return rule[1]

    def __repr__(self):
        """
         print the generated rules
        """

        for rule in self.rules:
            print("If", end="")
            count = 0
            featurevalues = rule[0]
            for featurevalue in featurevalues:
                feature = featurevalue[0]
                value = featurevalue[1]
                featurename = self.columnnames[feature]
                valuename = self.valueencoding[value]
                print(" " + featurename + " is " + valuename, end="")
                count = count + 1
                if (count == len(featurevalues)):
                    print(" then " "the class is " + self.valueencoding[rule[1]], end="")
                else:
                    print(" and ", end="")
            print()


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
    columnnames = ["outlook", "temperature", "humidity", "wind"]
    prism = Prism(value_encoding, columnnames)
    prism.fit(X, y)
    prism.__repr__()
    sample=[13,4,7,8]
    valueencoding = {value: key for key, value in value_encoding.items()}
    convertedsample= [valueencoding[value] for value in sample ]

    print("prediction for " + str(convertedsample) + " is " +  valueencoding[prism.predict(sample)])

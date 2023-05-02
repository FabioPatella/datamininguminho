import unittest
import numpy as np

from decisiontree import DecisionTree


class TreeTest(unittest.TestCase):
    def setUp(self):
        # Define the input features
        outlook = ['Overcast', 'Overcast', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Overcast', 'Rain',
                   'Sunny', 'Sunny', 'Rain', 'Sunny']
        temperature = ['Hot', 'Mild', 'Mild', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Hot', 'Cool', 'Hot', 'Hot',
                       'Mild',
                       'Mild']
        humidity = ['Normal', 'High', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'High', 'High', 'Normal',
                    'High',
                    'High', 'High', 'High']
        wind = ['Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
                'Weak',
                'Strong', 'Weak']

        # Define the target variable
        play_tennis = ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No']

        # Convert input features to numerical data
        outlook_num = np.array([12, 12, 13, 14, 13, 12, 14, 14, 12, 14, 13, 13, 14, 13])
        temperature_num = np.array([15, 4, 4, 4, 5, 5, 5, 4, 15, 5, 15, 15, 4, 4])
        humidity_num = np.array([6, 7, 6, 6, 6, 6, 6, 7, 7, 6, 7, 7, 7, 7])
        wind_num = np.array([8, 9, 9, 8, 8, 9, 8, 8, 8, 9, 9, 8, 9, 8])

        # Combine input features into X matrix
        self.X = np.vstack((outlook_num, temperature_num, humidity_num, wind_num)).T
        self.Xnames = np.vstack((outlook, temperature, humidity, wind))
        self.ynames = play_tennis

        # Convert target variable to numerical data
        play_tennis_num = np.array([11 if x == 'Yes' else 10 for x in play_tennis])

        # Define target variable y
        self.y = play_tennis_num
        self.value_encoding = {
            'Sunny': 13, 'Overcast': 12, 'Rain': 14, 'Hot': 15, 'Mild': 4, 'Cool': 5, 'Normal': 6
            , 'High': 7, 'Weak': 8,
            'Strong': 9, 'No': 10, 'Yes': 11, 'outlook': 0, 'temperature': 1, 'humidity': 2, 'wind': 3
        }
    def testentropy(self):
            decisiontree = DecisionTree()
            decisiontree.fit(self.X, self.y)
            self.assertEqual(decisiontree.get_tree(),{0: {12: 11, 13: {2: {6: 11, 7: 10}}, 14: {3: {8: 11, 9: 10}}}})
    def testgini(self):
        decisiontree = DecisionTree('gini')
        decisiontree.fit(self.X, self.y)
        self.assertEqual(decisiontree.get_tree(), {0: {12: 11, 13: {2: {6: 11, 7: 10}}, 14: {3: {8: 11, 9: 10}}}})

    def testgainratio(self):
        decisiontree = DecisionTree('gain_ratio')
        decisiontree.fit(self.X, self.y)
        self.assertEqual(decisiontree.get_tree(), {0: {12: 11, 13: {2: {6: 11, 7: 10}}, 14: {3: {8: 11, 9: 10}}}})
    def testpruningbydepth(self):
        decisiontree = DecisionTree(max_depth=1)
        decisiontree.fit(self.X, self.y)
        self.assertEqual(decisiontree.get_tree(), {0: {12: 11, 13: 10, 14: 11}})
    def testpredict(self):
        decisiontree = DecisionTree()
        decisiontree.fit(self.X, self.y)
        outlook_postpruning = np.array([12, 13, 14, 13])
        temperature_postpruning = np.array([4, 15, 4, 4])
        humidity_postpruning = np.array([7, 6, 6, 7])
        wind_postpruning = np.array([9, 9, 9, 8])
        y_postpruning = np.array([11, 11, 10, 10])
        X = np.vstack((outlook_postpruning, temperature_postpruning, humidity_postpruning, wind_postpruning)).T
        self.assertEqual(decisiontree.predict(X),[11,11,10,10])
    def test_reduce_error_pruning(self):
        decisiontree = DecisionTree()
        decisiontree.fit(self.X, self.y)
        outlook_postpruning = np.array([12, 13, 14, 13])
        temperature_postpruning = np.array([4, 15, 4, 4])
        humidity_postpruning = np.array([7, 6, 6, 7])
        wind_postpruning = np.array([9, 9, 9, 8])
        y_postpruning = np.array([11, 11, 10, 10])
        Xval = np.vstack((outlook_postpruning, temperature_postpruning, humidity_postpruning, wind_postpruning)).T
        decisiontree.reduce_error_pruning(Xval)
        self.assertEquals(decisiontree.get_tree(), {0: {12: 11, 13: {2: {6: 11, 7: 10}}, 14: 10}})


if __name__ == '__main__':
    unittest.main()

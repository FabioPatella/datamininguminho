import unittest
import numpy as np

from NaiveBayesClassifier import NaiveBayesClassifier
from NaiveBayesForText import NaiveBayesForText
from Prism import Prism


class BaiveBayesPrism(unittest.TestCase):
    def setUp(self):
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
        self.X=X
        self.y=y
        self.value_encoding=value_encoding
        self.column_names= ["outlook", "temperature", "humidity", "wind"]

    def testbayes(self):
            naiveBayesClassifier = NaiveBayesClassifier()
            column_names = ['outlook', 'temperature', 'humidity', 'wind']
            naiveBayesClassifier.fit(self.X, self.y, column_names)
            outlook_num = np.array([12, 12, 13, 14, 13, 12, 14, 14, 12, 14, 13, 13, 14, 13])
            temperature_num = np.array([15, 4, 4, 4, 5, 5, 5, 4, 15, 5, 15, 15, 4, 4])
            humidity_num = np.array([6, 7, 6, 6, 6, 6, 6, 7, 7, 6, 7, 7, 7, 7])
            wind_num = np.array([8, 9, 9, 8, 8, 9, 8, 8, 8, 9, 9, 8, 9, 8])

            # Combine input features into X matrix
            X = np.vstack((outlook_num, temperature_num, humidity_num, wind_num)).T
            self.assertEqual(naiveBayesClassifier.predict(X),[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1])

    def testPrism(self):
        prism = Prism(self.value_encoding, self.column_names)
        prism.fit(self.X, self.y)
        sample=sample=[13,4,7,8]
        self.assertEqual(prism.predict(sample),0)
    def testBaiyesFortext(self):
        text = [("a baixa do porto", "Porto"), ("o mercado do bolhão é no porto", "Porto"),
                ("a câmara do porto fica no centro do porto", "Porto"), ("a baixa de lisboa", "Lisboa"),
                ("o porto de lisboa", "Lisboa")
                ]
        nvt = NaiveBayesForText()
        nvt.fit(text, ["Porto", "Lisboa"])
        prediction = nvt.predict("Lisboa è bella")
        self.assertEqual(prediction,"Lisboa")


if __name__ == '__main__':
    unittest.main()

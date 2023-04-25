import numpy as np
from scipy.stats import f_oneway
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from Dataset import Dataset
import statsmodels.api as sm
from sklearn import linear_model
from statsmodels.api import OLS, add_constant
from scipy.stats import f
from sklearn.feature_selection import chi2

from fscores import f_classif, f_chi2, f_regression


# Select features based on the false discovery rate (FDR) procedure
class SelectFdr:
    def __init__(self, score_func):
        # Initialize the object with the score function to be used
        self.score_func = score_func

        # Initialize the F-statistic and p-value to None
        self.F_ = None
        self.p_ = None

    # Fit the transformer to the data and calculate the F-statistic and p-value
    def fit(self, dataset, y=None):
        # Call the score function on the dataset to get the F-statistic and p-value
        scores = self.score_func(dataset)
        self.F_, self.p_ = scores

        # Return the transformer object
        return self

    # Transform the dataset by selecting features based on the FDR procedure
    def transform(self, dataset):
        # Check if the transformer has been fitted before
        if self.p_ is None:
            raise ValueError("The transformer has not been fitted yet.")

        # Perform the Benjamini-Hochberg procedure on the p-values to control the FDR
        benjamini_hochberg(self.p_, dataset)

    # Fit the transformer to the data and transform it in one step
    def fit_transform(self, dataset, y=None):
        # Fit the transformer to the data
        self.fit(dataset, y)

        # Transform the data by selecting features based on the FDR procedure
        return self.transform(dataset, y)


def benjamini_hochberg(p_values,dataset, alpha=0.05):
        """
        Benjamini-Hochberg method for controlling the false discovery rate (FDR).

        :param p_values: A numpy array containing the p-values of the hypothesis tests.
        :param alpha: The desired significance level (default = 0.05).

        :return: A numpy array containing the Benjamini-Hochberg corrected p-values.
        """
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        features= np.array(dataset.features, dtype=str)
        features = features[sorted_indices]
        #dataset.features=features.tolist()
        dataset.x=[dataset.x[i] for i in sorted_indices]


        alpha_seq = np.arange(1, m + 1) / m * alpha   #This line calculates the sequence of critical values for the Benjamini-Hochberg method, where alpha is the significance level. The sequence is an array of length m, where each element is equal to i/m * alpha for i from 1 to m. The values in this sequence will be used to compare against the sorted p-values later on to determine the adjusted p-values

        # Find the largest index i for which p[i] <= alpha_seq[i]
        count=0
        for i in range(m - 1, -1, -1):
            count=count +1
            if sorted_p_values[i] <= alpha_seq[i]:
                break

        # Compute the number of rejected null hypotheses (k)
        k = count + -1

        #remove last k elements from inpit matrix and features
        features=features[:-k]
        dataset.features=features.tolist()
        dataset.x=dataset.x[:-k]




if __name__ == '__main__':

    #linear regression test
    print("linear regression selection:")
    X= [np.array([1.1,2,3,4,5]),np.array([2,4,6,8,10]),np.array([2,4,6,8,10]),np.array([2,4,6,8,10])]
    y = np.array([2, 4, 6, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2','feat3','feat4'], 'output')
    dataset.describe()
    selectfdr = SelectFdr(f_regression)
    selectfdr = selectfdr.fit(dataset)
    selectfdr.transform(dataset)
    dataset.describe()

    #anova test
    print("anova selection:")
    X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]), np.array([2, 4, 6, 8, 10])]
    y = np.array([2, 4, 4, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2','feat3'], 'output')
    dataset.describe()
    selectfdr = SelectFdr(f_classif)
    selectfdr = selectfdr.fit(dataset)
    selectfdr.transform(dataset)
    dataset.describe()
    # chi2 test
    print("chi2 selection:")
    X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])]
    y = np.array([2, 4, 4, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2'], 'output')
    dataset.describe()
    selectfdr = SelectFdr(f_chi2)
    selectfdr = selectfdr.fit(dataset)
    selectfdr.transform(dataset)
    dataset.describe()
    

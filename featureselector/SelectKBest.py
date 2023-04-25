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

from fscores import f_regression, f_classif, f_chi2


# Select the top k features based on a score function
class SelectKBest:
    def __init__(self, score_func, k):
        # Initialize the object with the score function and k value
        self.score_func = score_func
        self.k = k

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

    # Transform the dataset by selecting the top k features based on the F-statistic or p-value
    def transform(self, dataset):
        # Check if the transformer has been fitted before
        if self.p_ is None:
            raise ValueError("The transformer has not been fitted yet.")

        # Get the input matrix from the dataset
        X = dataset.getinputmatrix()

        # Get the indices of the top k features based on the score function
        idx = np.argsort(self.p_)[:self.k]

        # Select the top k features from the input matrix
        X = X[:, idx]

        # Update the dataset's features and input matrix with the selected features
        dataset.features = [dataset.features[index] for index in idx]
        dataset.setinputmatrix(X)

    # Fit the transformer to the data and transform it in one step
    def fit_transform(self, dataset, y=None):
        # Fit the transformer to the data
        self.fit(dataset, y)

        # Transform the data by selecting the top k features based on the F-statistic or p-value
        return self.transform(dataset, y)


if __name__ == '__main__':

    #linear regression test
    print("linear regression selection:")
    X= [np.array([1.1,2,3,4,5]),np.array([2,4,6,8,10])]
    y = np.array([2, 4, 6, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2'], 'output')
    dataset.describe()
    selectk = SelectKBest(f_regression, 1)
    selectk = selectk.fit(dataset)
    selectk.transform(dataset)
    dataset.describe()

    #anova test
    print("anova selection:")
    X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])]
    y = np.array([2, 4, 4, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2'], 'output')
    dataset.describe()
    selectk = SelectKBest(f_classif, 1)
    selectk = selectk.fit(dataset)
    selectk.transform(dataset)
    dataset.describe()
    #chi2 test
    print("chi2 selection:")
    X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])]
    y = np.array([2, 4, 4, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2'], 'output')
    dataset.describe()
    selectk = SelectKBest(f_chi2, 1)
    selectk = selectk.fit(dataset)
    selectk.transform(dataset)
    dataset.describe()




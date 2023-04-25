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


class SelectPercentile:
    def __init__(self, score_func, percentile):
        """
        Constructor for the SelectPercentile class.

        Parameters:
        score_func (function): The scoring function used to evaluate the features.
        percentile (float): The percentile of features to keep after evaluation.
        """
        self.score_func = score_func
        self.percentile = percentile
        self.F_ = None
        self.p_ = None

    def fit(self, dataset, y=None):
        """
        Fit the SelectPercentile instance on the given dataset.

        Parameters:
        dataset (Dataset): The dataset to fit the instance on.
        y (array-like): The target variable of the dataset.

        Returns:
        self (SelectPercentile): The fitted SelectPercentile instance.
        """
        scores = self.score_func(dataset)
        self.F_, self.p_ = scores
        return self

    def transform(self, dataset):
        """
        Transform the given dataset to select the top percentile of features based on the fitted scores.

        Parameters:
        dataset (Dataset): The dataset to transform.

        Returns:
        None
        """
        if self.p_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        X = dataset.getinputmatrix()
        k = int(self.percentile * dataset.x.__len__())
        if(k<1):
            k=1
        idx = np.argsort(self.p_)[:k]
        X = X[:, idx]
        dataset.features = [dataset.features[index] for index in idx]
        dataset.setinputmatrix(X)

    def fit_transform(self, X, y=None):
        """
        Fit and transform the given dataset to select the top percentile of features based on the scores.

        Parameters:
        X (Dataset): The dataset to fit and transform.
        y (array-like): The target variable of the dataset.

        Returns:
        The transformed dataset.
        """
        self.fit(X, y)
        return self.transform(X, y)





if __name__ == '__main__':
    #linear regression test
    print("linear regression selection:")
    X= [np.array([1.1,2,3,4,5]),np.array([2,4,6,8,10]),np.array([2,4,6,8,10]),np.array([2,4,6,8,10])]
    y = np.array([2, 4, 6, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2','feat3','feat4'], 'output')
    dataset.describe()
    selectp = SelectPercentile(f_regression, 0.5)
    selectp = selectp.fit(dataset)
    selectp.transform(dataset)
    dataset.describe()

    #anova test
    print("anova selection:")
    X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]), np.array([2, 4, 6, 8, 10])]
    y = np.array([2, 4, 4, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2','feat3'], 'output')
    dataset.describe()
    selectp = SelectPercentile(f_classif, 0.5)
    selectp = selectp.fit(dataset)
    selectp.transform(dataset)
    dataset.describe()

    # chi2 test
    print("chi2 selection:")
    X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])]
    y = np.array([2, 4, 4, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2'], 'output')
    dataset.describe()
    selectp = SelectPercentile(f_chi2,0.5)
    selectp = selectp.fit(dataset)
    selectp.transform(dataset)
    dataset.describe()
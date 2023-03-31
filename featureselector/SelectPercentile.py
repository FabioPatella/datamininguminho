import numpy as np
from scipy.stats import f_oneway
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from dataset import Dataset
import statsmodels.api as sm
from sklearn import linear_model
from statsmodels.api import OLS, add_constant
from scipy.stats import f


class SelectPercentile:
    def __init__(self, score_func, percentile):
        self.score_func = score_func
        self.percentile = percentile
        self.F_ = None
        self.p_ = None

    def fit(self, dataset, y=None):
        scores = self.score_func(dataset)
        self.F_, self.p_ = scores
        return self

    def transform(self,
                  dataset):
        if self.p_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        X= dataset.getinputmatrix()
        k=int(self.percentile * dataset.x.__len__())
        if(k<1):k=1
        idx = np.argsort(self.p_)[:k]
        X = X[:, idx]
        dataset.features = [dataset.features[index] for index in idx]
        dataset.setinputmatrix(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


def f_classif(dataset):
    X=dataset.getinputmatrix()
    y=dataset.y
    classes= np.unique(y)
    groups = [X[dataset.y == c] for c in classes]
    print(groups)
    F, p = f_oneway(*groups)
    print(p)
    return(F,p)


def f_regression(dataset):
    X=dataset.getinputmatrix()
    print(X)
    print(dataset.y)
    model = OLS(dataset.y, X).fit()
    return None, model.pvalues


if __name__ == '__main__':



    #linear regression test
    X= [np.array([1.1,2,3,4,5]),np.array([2,4,6,8,10]),np.array([2,4,6,8,10]),np.array([2,4,6,8,10])]
    y = np.array([2, 4, 6, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2','feat3','feat4'], 'output')
    dataset.describe()
    selectk = SelectPercentile(f_regression, 0.5)
    selectk = selectk.fit(dataset)
    selectk.transform(dataset)
    dataset.describe()

    #anova test
    X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]), np.array([2, 4, 6, 8, 10])]
    y = np.array([2, 4, 4, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2','feat3'], 'output')
    dataset.describe()
    selectk = SelectPercentile(f_classif, 0.5)
    selectk = selectk.fit(dataset)
    selectk.transform(dataset)
    dataset.describe()
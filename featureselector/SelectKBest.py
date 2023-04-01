import numpy as np
from scipy.stats import f_oneway
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from dataset import Dataset
import statsmodels.api as sm
from sklearn import linear_model
from statsmodels.api import OLS, add_constant
from scipy.stats import f
from sklearn.feature_selection import chi2


class SelectKBest:
    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
        self.F_ = None
        self.p_ = None

    def fit(self, dataset, y=None):
        scores = self.score_func(dataset)
        self.F_, self.p_ = scores
        return self

    def transform(self,
                  dataset):  # prende i pvalue piu piccoli e triene le colonne delle rispettive features,il resto lo scarta
        if self.p_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        X= dataset.getinputmatrix()
        idx = np.argsort(self.p_)[:self.k]
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
    F, p = f_oneway(*groups)
    return(F,p)


def f_regression(dataset):
    X=dataset.getinputmatrix()
    model = OLS(dataset.y, X).fit()
    return None, model.pvalues
def f_chi2(dataset):
    X=dataset.getinputmatrix()
    chi2_scores, p_values = chi2(X, dataset.y)
    return chi2_scores, p_values


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




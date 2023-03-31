import numpy as np
from scipy.stats import f_oneway
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from dataset import Dataset
import statsmodels.api as sm
from sklearn import linear_model
from statsmodels.api import OLS, add_constant
from scipy.stats import f


class SelectFdr:
    def __init__(self, score_func ):
        self.score_func = score_func
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
        benjamini_hochberg(self.p_,dataset)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
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
    selectk = SelectFdr(f_regression)
    selectk = selectk.fit(dataset)
    selectk.transform(dataset)
    dataset.describe()

    #anova test
    X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]), np.array([2, 4, 6, 8, 10])]
    y = np.array([2, 4, 4, 8, 10])
    dataset = Dataset(X, y, ['feat1', 'feat2','feat3'], 'output')
    dataset.describe()
    selectk = SelectFdr(f_classif)
    selectk = selectk.fit(dataset)
    selectk.transform(dataset)
    dataset.describe()
import numpy as np
from scipy.stats import f_oneway
from sklearn.feature_selection import chi2
from statsmodels.api import OLS


# Performs ANOVA test on the dataset
def f_classif(dataset):
    # Extract input matrix X and target variable y from dataset
    X = dataset.getinputmatrix()
    y = dataset.y

    # Get unique classes in y
    classes = np.unique(y)

    # Group the samples in X by class
    groups = [X[dataset.y == c] for c in classes]

    # Perform the F-test (ANOVA) on the groups
    F, p = f_oneway(*groups)

    # Return the F-statistic and p-value
    return F, p


# Performs linear regression test on the dataset
def f_regression(dataset):
    # Extract input matrix X and target variable y from dataset
    X = dataset.getinputmatrix()
    y = dataset.y

    # Fit a linear regression model
    model = OLS(y, X).fit()

    # Return the p-values of the coefficients
    return None, model.pvalues


# Performs chi-square test on the dataset
def f_chi2(dataset):
    # Extract input matrix X and target variable y from dataset
    X = dataset.getinputmatrix()
    y = dataset.y

    # Perform the chi-square test on X and y
    chi2_scores, p_values = chi2(X, y)

    # Return the chi-square scores and p-values
    return chi2_scores, p_values

import unittest

from numpy.testing import assert_array_equal

from SelectFdr import SelectFdr
from SelectKBest import SelectKBest
from Dataset import Dataset
from SelectPercentile import SelectPercentile

from fscores import f_regression, f_classif, f_chi2
import numpy as np


import unittest
import numpy as np
from statsmodels.api import OLS
from sklearn.feature_selection import f_oneway, chi2

from Dataset import Dataset
from variancethreshold import VarianceThreshold


class TestSelector(unittest.TestCase):
    def setUp(self):
        X = [np.array([1.1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])]
        y = np.array([2, 4, 6, 8, 10])
        self.dataset = Dataset(X, y, ['feat1', 'feat2'], 'output')

    def test_Kbest_regr(self):
        selectk = SelectKBest(f_regression, 1)
        selectk = selectk.fit(self.dataset)
        selectk.transform(self.dataset)
        totest= [np.array([2,4,6,8,10])]
        for index in range(self.dataset.x.__len__()):
          self.assertEqual(self.dataset.x[index].tolist(),totest[index].tolist())

    def testPercentile_fclassif(self):
        selectp = SelectPercentile(f_classif, 0.5)
        selectp = selectp.fit(self.dataset)
        selectp.transform(self.dataset)
        totest = [np.array([1.1, 2.0, 3.0, 4.0, 5.0])]
        for index in range(self.dataset.x.__len__()):
            self.assertEqual(self.dataset.x[index].tolist(), totest[index].tolist())
    def testFdr_chi(self):
        selectfdr = SelectFdr(f_chi2)
        selectfdr = selectfdr.fit(self.dataset)
        selectfdr.transform(self.dataset)
        totest = [np.array([2, 4, 6, 8, 10])]
        for index in range(self.dataset.x.__len__()):
            self.assertEqual(self.dataset.x[index].tolist(), totest[index].tolist())
    def variancethreshold(self):
        vt = VarianceThreshold(0.10)
        vt = vt.fit(self.dataset)
        self.dataset = vt.transform(self.dataset)
        totest = [np.array([2, 4, 6, 8, 10])]
        for index in range(self.dataset.x.__len__()):
            self.assertEqual(self.dataset.x[index].tolist(), totest[index].tolist())



if __name__ == '__main__':
    unittest.main()






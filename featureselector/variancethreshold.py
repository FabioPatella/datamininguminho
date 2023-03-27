import os
from dataset import Dataset
import numpy as np




class VarianceThreshold:

    def __init__(self, threshold: float = 0.0):
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        self.threshold = threshold
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        self.variance=[]
        self.features=dataset.features
        for indexrow in range(dataset.x.__len__()):
             try:
              self.variance.append(np.var(dataset.x[indexrow]))
             except:
              self.variance.append(0.0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        X=[]
        features=[]
        for index in range(dataset.x.__len__()):
            if(self.variance[index]<=self.threshold):
                X.append(dataset.x[index])
                features.append(dataset.features[index])


        return Dataset(X, dataset.y, features, dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    firstrow = np.array([1, 1, 2])
    secondrow = np.array(['a', None, 'c'])
    thirdrow = np.array([1, 1, 1])
    x = [firstrow, secondrow, thirdrow]
    dataset = Dataset(x, np.array([1, 2, 3]), ['feat1', 'feat2', 'feat3'], 'output')
    dataset.describe()
    print("after variance threshold: ")
    vt= VarianceThreshold(0.10)
    vt= vt.fit(dataset)
    dataset= vt.transform(dataset)
    dataset.describe()


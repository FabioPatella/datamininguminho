# This is a sample Python script.
from typing import Sequence, Tuple

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd

import csv
import numpy as np


class Dataset:
    def __init__(self, x: Sequence[np.ndarray], y: np.ndarray = None, features: Sequence[str] = None,
                 label: str = None):
        if x is None:
            raise ValueError("X cannot be None")
        if x.__len__()==0:
            raise ValueError("X cannot be empty")

        if features is None:
            features = [str(i) for i in range(x.shape[1])]
        else:
            features = list(features)

        if y is not None and label is None:
            label = "y"

        self.x = x
        self.y = y
        self.features = features
        self.label = label

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def set_x(self, new_x):
        self.x = new_x

    def set_y(self, new_y):
        self.y = np.array(new_y)

    def read_csvtsv(self, filename, delimiter=",", dataformat='csv'):
        if dataformat == "tsv": delimiter = "\t"
        self.x = []
        self.y = np.array([])
        self.label = None
        self.features = []

        with open(filename) as csv_file:
            data = []
            reader = csv.reader(csv_file, delimiter=delimiter)
            firstrow = True
            for row in reader:
                if firstrow:
                    self.features = row[:-1]
                    self.label = row[-1]
                    firstrow = False
                    for index in range(row.__len__() - 1):
                        self.x.append(np.array([]))

                else:
                    for index in range(row.__len__()):
                        if index != row.__len__() - 1:
                            self.x[index] = np.append(self.x[index], row[index])

                        else:
                            self.y = np.append(self.y, row[index])

    def write_csvtsv(self,filename, delimiter=",",dataformat='csv'):
        if dataformat == "tsv": delimiter = "\t"
        with open(filename, mode='w', newline='',) as csv_file:
            writer = csv.writer(csv_file, delimiter=delimiter)
            # scriviamo la riga delle features e della label
            row = self.features + [self.label]
            writer.writerow(row)

            # scriviamo i dati
            for indexrow in range(self.x[0].__len__()):
                row=[]
                for indexcolumn in range(self.x.__len__()):
                    row.append(self.x[indexcolumn][indexrow])
                row.append(self.y[indexrow])
                writer.writerow(row)





    def shape(self):
        return (self.x[0].__len__(), self.x.__len__())

    def describe(self):
        print("Shape:(", self.x[0].__len__(), ",", self.x.__len__(), ")")
        for index in range(self.shape()[1]):
            print(self.features[index], " of type ", type(self.x[index][0]), end="")
            print(" values: ", end="")
            for indexcolumn in range(self.shape()[0]):
                if indexcolumn < self.shape()[0] - 1:
                    print(self.x[index][indexcolumn], ",", end="")
                else:
                    print(self.x[index][indexcolumn], end="")
            print()
        print("output label :", self.label, end="")
        print(" ", self.y)

    def count_null_values(self):
        for indexrow in range(self.x.__len__()):
            columncount=0
            for indexcolumn in range(self.x[0].__len__()):
                if(self.x[indexrow][indexcolumn] is None): columncount=columncount+1
            print(self.features[indexrow]," null values are ",columncount)


    def replace_null_values(self, method='commonvalues'):
        if method == 'commonvalues':
         for index in range(self.x.__len__()):
             dictionary={}
             for indexcolumn in range(self.x[0].__len__()):
                 if self.x[index][indexcolumn] in dictionary:
                   dictionary[self.x[index][indexcolumn]] = dictionary[self.x[index][indexcolumn]] + 1
                 else:
                   dictionary[self.x[index][indexcolumn]] = 1
             if None in dictionary: del dictionary[None]
             key_with_max_value = max(dictionary, key=dictionary.get)
             for indexcolumn in range(self.x[index].__len__()):
                 if self.x[index][indexcolumn] is None: self.x[index][indexcolumn] = key_with_max_value


             #for indexcolumn in range(self.x[index].__len__()):




        elif method == 'mean':
            for index in range(self.x.__len__()):
              try:
                   float_arr= self.x[index].astype(float)
                   arr_without_nan = float_arr[np.logical_not(np.isnan(float_arr))]
                   mean = np.mean(arr_without_nan)
                   arr_type= type(self.x[index][0])
                   mean=arr_type(mean)
                   for indexcolumn in range(self.x[index].__len__()):
                       if self.x[index][indexcolumn] is None: self.x[index][indexcolumn] = mean
              except Exception as e:

                       print(f"{str(e)} ({type(e)})")



        else:
            print("Invalid method. Options: mode, mean")




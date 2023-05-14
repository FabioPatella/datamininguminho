# This is a sample Python script.
from typing import Sequence, Tuple

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd

import csv
import numpy as np


class Dataset:
    def __init__(self, x: Sequence[np.ndarray]=None, y: np.ndarray = None, features: Sequence[str] = None,
                 label: str = None,filename=None):
        # constructor for Dataset class
        # x: input features as a list of np.ndarrays
        # y: output labels as a np.ndarray
        # features: list of feature names
        # label: name of output label
        if filename is not None:
            self.readDataset(filename)
            return
        if x is None:
            raise ValueError("X cannot be None")
        if x.__len__()==0:
            raise ValueError("X cannot be empty")

        if features is None:
            features = [str(i) for i in range(x.__len__())]
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
    def readDataset(self, filename, sep = ","):
        data = np.genfromtxt(filename, delimiter=sep)
        X = data[:,0:-1]
        Y = data[:,-1]
        self.setinputmatrix(X)
        X2=self.getinputmatrix()
        self.y=Y
        features=[]
        for index in range(X.shape[1]):
            features.append("feat" + str(index))
        self.features=features
        self.label="y"



    def read_csvtsv(self, filename, delimiter=",", dataformat='csv'):
        """
        # method to read dataset from a csv/tsv file
        # filename: path to file
        # delimiter: delimiter to use for parsing file
        # dataformat: csv/tsv
        """
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
        """
        Writes in a file the input matrix of the dataset
        :param filename: name of the file
        :param delimiter: delimiter to use for writing file
        :param dataformat: string to write in csv or tsv file
        :return:
        """
        if dataformat == "tsv": delimiter = "\t"
        with open(filename, mode='w', newline='',) as csv_file:
            writer = csv.writer(csv_file, delimiter=delimiter)
            row = self.features + [self.label]
            writer.writerow(row)
            for indexrow in range(self.x[0].__len__()):
                row=[]
                for indexcolumn in range(self.x.__len__()):
                    row.append(self.x[indexcolumn][indexrow])
                row.append(self.y[indexrow])
                writer.writerow(row)





    def shape(self):
        """
        :return: the input matrix shape
        """
        return (self.x[0].__len__(), self.x.__len__())

    def describe(self):
        """
        print in the console a full description of the dataset, including shape,columns,input matrix values and output values with label
        """
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
        """
        Count the null values in all the dataset and print them in the console
        """
        for indexrow in range(self.x.__len__()):
            columncount=0
            for indexcolumn in range(self.x[0].__len__()):
                if(self.x[indexrow][indexcolumn] is None): columncount=columncount+1
            print(self.features[indexrow]," null values are ",columncount)

    def getinputmatrix(self):
     return np.vstack(self.x).T
    def setinputmatrix(self,X):

        self.x=np.hsplit(X, 1)
    def replace_null_values(self, method='commonvalues'):
        """
        Replace null values with commob values or with mean of values
        :param method: if commonvalues the method replace null values with the most common value for each column , otherwise if it is mean it replaces null values with the mean
        :return: doesn't return anything,it just modifies the dataset object on which the method is called
        """
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




    def select_rows_by_position(self, positions: Sequence[int]) -> 'Dataset':
            """
            Select rows based on their positions in the dataset

            :param positions: List of row indices to select
            :return: A new dataset object with only the selected rows
            """
            new_x = [self.x[i] for i in positions]
            new_y = self.y[positions] if self.y is not None else None
            return Dataset(new_x, new_y, self.features, self.label)

    def select_columns_by_position(self, positions: Sequence[int]) -> 'Dataset':
            """
            Select columns based on their positions in the dataset

            :param positions: List of column indices to select
            :return: A new dataset object with only the selected columns
            """
            new_x = [self.x[i] for i in positions]
            new_features = [self.features[i] for i in positions]
            return Dataset(new_x, self.y, new_features, self.label)

    def sort_by_feature(self, feature_name, ascending=True):
        """
        Sort columns
        :param feature_name: column to sort
        :param ascending: order to sort data
        :return: A new dataset object with the specified column sorted
        """
        index = self.features.index(feature_name)
        order = np.argsort(self.x[index])
        if not ascending:
            order = np.flip(order)
        new_x = [x[order] for x in self.x]
        new_y = self.y[order] if self.y is not None else None
        return Dataset(new_x, new_y, self.features, self.label)

    def remove_features(self, feature_names):
        """
                Remove features from the dataset
                :param feature_name: list of feature to remove
                :return: A new dataset object without the specified columns
                """
        new_features = [feat for feat in self.features if feat not in feature_names]
        new_x = []
        for i in range(len(self.features)):
            if self.features[i] not in feature_names:
                new_x.append(self.x[i])
        new_y = self.y if self.y is not None else None
        return Dataset(new_x, new_y, new_features, self.label)

    def filter_dataset(self,column_name, value, operator='='):
        """
        Filters a dataset based on a specified condition.

        Parameters:
        dataset (list): The input dataset.
        column_name (str): The name of the column to filter on.
        value (float): The value to compare the column values against.
        operator (str): The comparison operator to use. Default is '='.

        Returns:
        list: The filtered dataset.
        """
        X=self.getinputmatrix()
        column_index=self.features.index(column_name)
        Y=[]
        filtered_X = []
        rowindex=0
        for row in X:
            if operator == '=':
                if row[column_index] == value:
                    toappend= np.array(row)
                    filtered_X.append(toappend)
                    Y.append(self.y[rowindex])
            elif operator == '>':
                if row[column_index] > value:
                    toappend = np.array(row)
                    filtered_X.append(toappend)
                    Y.append(self.y[rowindex])
            elif operator == '<':
                if row[column_index] < value:
                    toappend = np.array(row)
                    filtered_X.append(toappend)
                    Y.append(self.y[rowindex])
            rowindex=rowindex+1
        newX=[]
        for indexcolumn in range(filtered_X[0].__len__()):
            column=[]
            for indexrow in range(filtered_X.__len__()):
                column.append(filtered_X[indexrow][indexcolumn])

            newX.append(np.array(column))

        dataset = Dataset(newX,np.array(Y),self.features,self.label)
        return dataset
    def train_test_split(self, p = 0.7):
        from random import shuffle
        X=self.getinputmatrix()
        numberofinstances = X.shape[0]
        inst_indexes = np.array(range(numberofinstances))
        numbertotraining = (int)(p*numberofinstances)
        shuffle(inst_indexes)
        tr_indexes = inst_indexes[0:numbertotraining]
        tst_indexes = inst_indexes[numbertotraining:]
        Xtr = X[tr_indexes,]
        ytr = self.y[tr_indexes]
        Xts = X[tst_indexes,]
        yts = self.y[tst_indexes]
        return (Xtr, ytr, Xts, yts)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    firstcolumn = np.array([1, 1, 2])
    secondcolumn = np.array(['a', None, 'c'])
    thirdcolumn = np.array([True, None, False])
    x = [firstcolumn, secondcolumn, thirdcolumn]
    dataset = Dataset(x, np.array([1, 2, 3]), ['feat1', 'feat2', 'feat3'], 'output')
    dataset.describe()
    xtr,ytr,xts,yts= dataset.train_test_split()
    print("x training\n",xtr)
    print("y training\n",ytr)
    print("x testing\n" ,xts)
    print("y testing\n",yts)
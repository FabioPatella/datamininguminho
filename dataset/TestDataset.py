import csv
import unittest
import numpy as np
from typing import List
import unittest
import numpy as np
from typing import List
from Dataset import Dataset
import pytest


class TestDataset(unittest.TestCase):




        def setUp(self):
            self.x = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
            self.y = np.array([0, 1, 0])
            self.features = ['feature1', 'feature2', 'feature3']
            self.label = 'target'
            self.dataset = Dataset(x=self.x, y=self.y, features=self.features, label=self.label)

        def test_init(self):
            # Test that the constructor works correctly
            self.assertIsInstance(self.dataset.x, List)
            self.assertIsInstance(self.dataset.y, np.ndarray)
            self.assertIsInstance(self.dataset.features, List)
            self.assertIsInstance(self.dataset.label, str)
            self.assertEqual(self.dataset.x, self.x)
            self.assertEqual(self.dataset.y.tolist(), self.y.tolist())
            self.assertEqual(self.dataset.features, self.features)
            self.assertEqual(self.dataset.label, self.label)

        def test_get_x(self):
            # Test that the get_x method works correctly
            self.assertEqual(self.dataset.get_x(), self.x)

        def test_get_y(self):
            # Test that the get_y method works correctly
            self.assertEqual(self.dataset.get_y().tolist(), self.y.tolist())

        def test_set_x(self):
            # Test that the set_x method works correctly
            new_x = [np.array([10, 20, 30]), np.array([40, 50, 60]), np.array([70, 80, 90])]
            self.dataset.set_x(new_x)
            self.assertEqual(self.dataset.get_x(), new_x)

        def test_set_y(self):
            # Test that the set_y method works correctly
            new_y = np.array([1, 0, 1])
            self.dataset.set_y(new_y)
            self.assertEqual(self.dataset.get_y().tolist(), new_y.tolist())


        def test_shape(self):
            # Test that the shape method works correctly
            self.assertEqual(self.dataset.shape(), (3, 3))



        def test_count_null_values_with_null_values(self):
            # Arrange
            x = np.array([
                [1, 2, None],
                [4, 5, 6]
            ])
            y = np.array([1, 2])
            features = ["Feature 1", "Feature 2", "Feature 3"]
            dataset = Dataset(x, y, features, "Label")

            # Act
            dataset.count_null_values()

            # Assert
            # expect output in console:
            # Feature 1 null values are 0
            # Feature 2 null values are 0
            # Feature 3 null values are 1


        def test_replace_null_values_with_common_values(self):
                # Arrange
                x = np.array([
                    [1, 2, None],
                    [1, 5, 6]
                ])
                y = np.array([1, 2])
                features = ["Feature 1", "Feature 2", "Feature 3"]
                dataset = Dataset(x, y, features, "Label")

                # Act
                dataset.replace_null_values('commonvalues')

                # Assert
                # expected x array:
                # [[1, 2, 1],
                #  [1, 5, 6]]
                assert np.array_equal(dataset.x, np.array([[1, 2, 1], [1, 5, 6]]))

        def test_replace_null_values_with_mean(self):
            # Arrange
            x = np.array([
                [1, 3, None],
                [4, None, 6]
            ])
            y = np.array([1, 2])
            features = ["Feature 1", "Feature 2", "Feature 3"]
            dataset = Dataset(x, y, features, "Label")

            # Act
            dataset.replace_null_values('mean')

            # Assert
            # expected x array:
            # [[1, 2, 3],
            #  [4, 2, 6]]
            assert np.array_equal(dataset.x, np.array([[1, 3, 2], [4, 5, 6]]))



        def test_readwrite_csvtsv(self):
            # Test that the write_csvtsv method works correctly
            filename = 'test.csv'
            delimiter = ','
            dataformat = 'csv'

            X = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
            Y = np.array([0, 1, 0])
            features = ['feature1', 'feature2', 'feature3']
            label = 'target'
            dataset = Dataset(X,Y, features=features, label=label)
            self.dataset.write_csvtsv(filename, delimiter=delimiter, dataformat=dataformat)
            dataset.read_csvtsv(filename, delimiter=delimiter, dataformat=dataformat)
            for indexcolumn in range (dataset.x.__len__()):
                for indexrow in range(dataset.x[0].__len__()):
                    elem=int(dataset.x[indexrow][indexcolumn])
                    self.assertEqual(elem,self.dataset.x[indexrow][indexcolumn])

            for index in range(len(dataset.y)):
                elem=int(dataset.y[index])
                self.assertEqual(elem,self.dataset.y[index])
            self.assertEqual(self.dataset.features, dataset.features)
            self.assertEqual(self.dataset.label, dataset.label)

        def test_select_rows_by_position(self):
            # Test selecting all rows
            new_dataset = self.dataset.select_rows_by_position([0, 1, 2])
            self.assertEqual(new_dataset.x, self.x)
            self.assertEqual(new_dataset.y.tolist(), self.y.tolist())
            self.assertEqual(new_dataset.features, self.features)
            self.assertEqual(new_dataset.label, self.label)

            # Test selecting a subset of rows
            new_dataset = self.dataset.select_rows_by_position([0, 2])
            self.assertEqual(new_dataset.x[0][0],1)
            self.assertEqual(new_dataset.x[1][0],7)
            self.assertEqual(new_dataset.y.tolist(), [0, 0])
            self.assertEqual(new_dataset.features, self.features)
            self.assertEqual(new_dataset.label, self.label)

        def test_select_columns_by_position(self):
            # Test selecting all columns
            new_dataset = self.dataset.select_columns_by_position([0, 1, 2])
            self.assertEqual(new_dataset.x, self.x)
            self.assertEqual(new_dataset.y.tolist(), self.y.tolist())
            self.assertEqual(new_dataset.features, self.features)
            self.assertEqual(new_dataset.label, self.label)

            # Test selecting a subset of columns
            new_dataset = self.dataset.select_columns_by_position([0, 2])
            self.assertEquals(new_dataset.x[0][0],1)
            self.assertEquals(new_dataset.x[1][1], 8)
            self.assertEqual(new_dataset.y.tolist(), self.y.tolist())
            self.assertEqual(new_dataset.features, ['feature1', 'feature3'])
            self.assertEqual(new_dataset.label, self.label)

        def test_sort_by_feature(self):
            # Test sorting in ascending order
            new_dataset = self.dataset.sort_by_feature('feature2')
            self.assertEqual(new_dataset.x[0][0],1)
            self.assertEqual(new_dataset.y.tolist(), [0, 1, 0])
            self.assertEqual(new_dataset.features, self.features)
            self.assertEqual(new_dataset.label, self.label)

            # Test sorting in descending order
            new_dataset = self.dataset.sort_by_feature('feature2', ascending=False)
            self.assertEqual(new_dataset.x[0][0],3)
            self.assertEqual(new_dataset.y.tolist(), [0, 1, 0])
            self.assertEqual(new_dataset.features, self.features)
            self.assertEqual(new_dataset.label, self.label)
        def test_remove_features(self):
            new_dataset= self.dataset.remove_features("feature1")
            self.assertEqual(new_dataset.x[0][0],4)
            self.assertEquals(new_dataset.features[0],"feature2")






if __name__ == '__main__':
    unittest.main()

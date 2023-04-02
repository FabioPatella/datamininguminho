import numpy as np


class DecisionTree:
    def __init__(self, criterion='entropy', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None


    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def get_tree(self):
        return self.tree

    def _build_tree(self, X, y, depth=0):
        number_samples, number_features = X.shape
        classes = np.unique(y)
        number_classes = len(classes)

        # Check termination conditions
        if number_classes == 1:
            return classes[0]
        if number_samples < 2 or depth == self.max_depth:
            return self._resolve_conflict(y)

        # Select best attribute
        info_gains = []
        for feature in range(number_features):
            info_gain = self._information_gain(X, y, feature)
            info_gains.append(info_gain)
        best_feature_index = np.argmax(info_gains)

        # Build subtrees
        tree = {best_feature_index: {}}
        for value in np.unique(X[:, best_feature_index]):
            mask = X[:, best_feature_index] == value
            sub_X = X[mask]  #all rows for which mask is true are assigned to sub_x
            sub_y = y[mask]
            sub_tree = self._build_tree(sub_X, sub_y, depth=depth + 1)
            tree[best_feature_index][value] = sub_tree

        return tree

    def _predict(self, x, tree):
        if isinstance(tree, np.ndarray):
            return tree
        feature = next(iter(tree))
        value = x[feature]
        if value not in tree[feature]:
            return self._resolve_conflict(list(tree[feature].values()))
        sub_tree = tree[feature][value]
        return self._predict(x, sub_tree)

    def _information_gain(self, X, y, feature):
        if self.criterion == 'entropy':
            parent_entropy = self._entropy(y)
        elif self.criterion == 'gini':
            parent_entropy = self._gini(y)
        elif self.criterion == 'gain_ratio':
            parent_entropy = self._entropy(y)
            intrinsic_value = self._intrinsic_value(X, feature)
            if intrinsic_value == 0:
                return 0
            return (parent_entropy - self._entropy_gain(X, y, feature)) / intrinsic_value

        n_samples = len(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        children_entropy = 0
        for value, count in zip(values, counts):
            mask = X[:, feature] == value
            child_y = y[mask]
            child_entropy = self._entropy(child_y) if self.criterion == 'entropy' else self._gini(child_y)
            children_entropy += count / n_samples * child_entropy
        return parent_entropy - children_entropy

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-6))

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _entropy_gain(self, X, y, feature):
        n_samples = len(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        children_entropy = 0
        for value, count in zip(values, counts):
            mask = X[:, feature] == value
            child_y = y[mask]
            child_entropy = self._entropy(child_y)
            children_entropy += count / n_samples * child_entropy
        return children_entropy

    def _intrinsic_value(self, X, feature):
        n_samples = len(X)
        values, counts = np.unique(X[:, feature], return_counts=True)
        iv = 0
        for value, count in zip(values, counts):
            probability = count / n_samples
            iv -= probability * np.log2(probability + 1e-6)
        return iv

    def _resolve_conflict(self, values):
        counts = np.bincount(values)
        return np.argmax(counts)

def view_tree(tree,embedded_values):
     easy_encoding= {value: key for key, value in embedded_values.items()}
     count=0
     if tree is not None :
         for key in tree.keys():
             if count==0: print('{',end='')
             count=count+1
             print(easy_encoding[key],end='')

             print(": ",end='')
             value= tree[key]
             if isinstance(value,dict):
                 view_tree(value,embedded_values)
                 print('}',end='')
             else:

                 print(easy_encoding[value],end='')

             if count != tree.keys().__len__():print(",",end='')








# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Define the input features
    outlook = ['Overcast', 'Overcast', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Overcast', 'Rain',
               'Sunny', 'Sunny', 'Rain', 'Sunny']
    temperature = ['Hot', 'Mild', 'Mild', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Hot', 'Cool', 'Hot', 'Hot', 'Mild',
                   'Mild']
    humidity = ['Normal', 'High', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'High', 'High', 'Normal', 'High',
                'High', 'High', 'High']
    wind = ['Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak',
            'Strong', 'Weak']

    # Define the target variable
    play_tennis = ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No']

    # Convert input features to numerical data
    outlook_num = np.array([12, 12, 13, 14, 13, 12, 14, 14, 12, 14, 13, 13, 14, 13])
    temperature_num = np.array([15, 4, 4, 4, 5, 5, 5, 4, 15, 5, 15, 15, 4, 4])
    humidity_num = np.array([6, 7, 6, 6, 6, 6, 6, 7, 7, 6, 7, 7, 7, 7])
    wind_num= np.array([8, 9, 9, 8, 8, 9, 8, 8, 8, 9, 9, 8, 9, 8])

    # Combine input features into X matrix
    X = np.vstack((outlook_num, temperature_num, humidity_num, wind_num)).T
    Xnames=np.vstack((outlook,temperature,humidity,wind))
    ynames=play_tennis

    # Convert target variable to numerical data
    play_tennis_num = np.array([11 if x == 'Yes' else 10 for x in play_tennis])

    # Define target variable y
    y = play_tennis_num
    value_encoding = {
        'Sunny': 13, 'Overcast': 12, 'Rain': 14, 'Hot': 15, 'Mild': 4, 'Cool': 5, 'Normal': 6
        , 'High': 7, 'Weak': 8,
        'Strong': 9, 'No': 10, 'Yes': 11, 'outlook': 0, 'temperature': 1, 'humidity': 2, 'wind': 3
    }
    # testing entropy
    print("entropy decision tree:")
    decisiontree= DecisionTree()
    decisiontree.fit(X,y)

    print(decisiontree.get_tree())
    view_tree(decisiontree.get_tree(),value_encoding)
    #testing gini index
    print()
    print("gini decision tree:")
    decisiontree = DecisionTree('gini')
    decisiontree.fit(X, y)
    print(decisiontree.get_tree())
    view_tree(decisiontree.get_tree(), value_encoding)
    # testing gain ratio
    print()
    print("gain ratio decision tree:")
    decisiontree = DecisionTree('gain_ratio')
    decisiontree.fit(X, y)
    print(decisiontree.get_tree())
    view_tree(decisiontree.get_tree(), value_encoding)

    #pre pruning by depth
    print()
    print("pre pruning by depth=1:")
    decisiontree = DecisionTree(max_depth=1)
    decisiontree.fit(X, y)
    print(decisiontree.get_tree())
    view_tree(decisiontree.get_tree(), value_encoding)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
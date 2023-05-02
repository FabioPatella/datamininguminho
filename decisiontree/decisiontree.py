import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

from DTNode import DTNode


class DecisionTree:
    def __init__(self, criterion='entropy', max_depth=None,prepruningindependence=False,confidence=0.05):
        """
                Initialize a DecisionTree object with the given parameters.

                Args:
                - criterion: the splitting criterion to use. Default is 'entropy'.
                - max_depth: the maximum depth of the decision tree. Default is None.
                """
        self.puretree = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.prepruningindependence=prepruningindependence
        self.confidence=confidence


    def fit(self, X, y):
        """
                Build a decision tree based on the input data X and labels y.

                Args:
                - X: input data, a numpy array of shape (number of samples, number of features).
                - y: labels, a numpy array of shape (number of samples,).

                Returns:
                - None
                """
        self.tree = self._build_tree(X, y)

    def predict(self, X,tree=None):
        """
               Predict the label of each sample in X using the trained decision tree.

               Args:
               - X: input data, a numpy array of shape (number of samples, number of features).

               Returns:
               - y_pred: predicted labels, a numpy array of shape (number of samples,).
               """
        results=[]
        for sample in X:
            if tree is not None: results.append(self._predict(sample, tree))
            else: results.append(self._predict(sample,self.tree))
        return results
    def _predict(self,sample,tree):
        feature = list(tree)[0]  # getting the feature
        value = sample[feature]
        subtree = tree[feature]
        result = subtree[value]
        if isinstance(result, dict): return self._predict(sample,result)
        else: return result

    def get_tree(self):
        return self.tree
    def convertdict(self):
        """
        obtain an equivalent representation of the tree with nodes class starting from a dictionary
        """
        feature = list(self.tree.keys())[0]
        self.puretree= DTNode(feature)
        for value in self.tree[feature].keys():
            newnode=DTNode(value)
            self.puretree.add_next(newnode)
            newnode.set_previous(self.puretree)
            self._updateconversionfromdict(newnode,self.tree[feature][value])

    def _updateconversionfromdict(self,node_,dictionary):
        """
        recursive private method to bui9ld nodes tree starting from a dictionary
        :param node_: current node
        :param dictionary:  current dictionary
        :return: a sub dictionary( in the end the final dictionary)
        """
        if  isinstance(dictionary, dict):
            feature = list(dictionary)[0]
            newnode_ = DTNode(feature)
            node_.add_next(newnode_)
            newnode_.set_previous(node_)
            #newnode_.set_previous(node_.get_previous())
            for value in dictionary[feature].keys():
                newnodeforvalue=DTNode(value)
                newnodeforvalue.set_previous(newnode_)
                newnode_.add_next(newnodeforvalue)
                self._updateconversionfromdict(newnodeforvalue,dictionary[feature][value])

        else:
            newnode_=DTNode(dictionary)
            node_.add_next(newnode_)
            newnode_.set_leaf(True)
            newnode_.set_previous(node_)
            #newnode_.set_previous(node_.get_previous())

    def convertpuretreetodict(self,currentnode):
        """
        convert the nodes tree into a dictionary
        """
        if(currentnode.isLeaf()): return currentnode.get_value()
        else:
         dict={}
         feature=currentnode.get_value()
         dict[feature]={}
         for child in currentnode.get_next():
             if(len(child.get_next())>1):
              dict[feature][child.get_value()]=self.convertpuretreetodict(child.get_next()[0])

             else:     dict[feature][child.get_value()]=self.convertpuretreetodict(child.get_next()[0])

         return dict














    def reduce_error_pruning(self,validation_set):
        """
        remove the deepest leaf in the tree until the performance of the new model get worse
        :param validation_set: validation data
        """
        self.convertdict()
        predictions= self.predict(validation_set)
        newpredictions=predictions
        while True:
         if predictions != newpredictions:
             self.tree=dict
             self.convertdict()
             break
         else:
          dict=self.convertpuretreetodict(self.puretree)
          node,deapth=self.find_deapest_leaf(self.puretree,0)
          father=node.get_previous()
          grandfather=father.get_previous()
          grand_grandfather=grandfather.get_previous()
          grand_grandfather.reset_next()
          grand_grandfather.add_next(node)
          newdict=self.convertpuretreetodict(self.puretree)
          newpredictions= self.predict(validation_set,newdict)





    def find_deapest_leaf(self,node,deapth):
        """
         returns the deepest leaf in a node tree
        :param node: current node
        :param deapth: current deapth
        :return: leaf and deapth
        """
        if(node.isLeaf()): return node,deapth+1
        else:
         max=0
         bestnode=None
         for child in node.get_next():
            candidatenode,child_deapth= self.find_deapest_leaf(child,deapth+1)
            if child_deapth>=max:
                max=child_deapth
                bestnode=candidatenode
         return bestnode,child_deapth

    def _build_tree(self, X, y, depth=0):
        """
                Build a decision tree recursively using the given data and labels.

                Args:
                - X: input data, a numpy array of shape (number of samples, number of features).
                - y: labels, a numpy array of shape (number of samples,).
                - depth: the current depth of the tree. Default is 0.

                Returns:
                - tree: a dictionary representing the decision tree.
                """
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
        if self.prepruningindependence:
            if(chi2_test(X[:, best_feature_index],y))>self.confidence:
             return self._resolve_conflict(y)

        # Build subtrees
        tree = {best_feature_index: {}}
        for value in np.unique(X[:, best_feature_index]):
            mask = X[:, best_feature_index] == value
            sub_X = X[mask]  #all rows for which mask is true are assigned to sub_x
            sub_y = y[mask]
            sub_tree = self._build_tree(sub_X, sub_y, depth=depth + 1)
            tree[best_feature_index][value] = sub_tree

        return tree



    def _information_gain(self, X, y, feature):
        # Calculate the information gain of a given feature
        if self.criterion == 'entropy':
            parent_entropy = self._entropy(y)
        elif self.criterion == 'gini':
            parent_entropy = self._gini(y)
        elif self.criterion == 'gain_ratio':
            parent_entropy = self._entropy(y)
            intrinsic_value = self._intrinsic_value(X, feature)
            if intrinsic_value == 0:
                return 0
            return (parent_entropy - self._entropy_gain(X, y, feature)) / intrinsic_value # Calculate information gain using entropy gain and intrinsic value

        n_samples = len(y) # Total number of samples in the dataset
        values, counts = np.unique(X[:, feature], return_counts=True) # Find unique feature values and their counts
        children_entropy = 0  # Initialize children entropy
        for value, count in zip(values, counts): # Loop over each unique feature value
            mask = X[:, feature] == value # Create a boolean mask for the samples having this value
            child_y = y[mask] # Get corresponding target values for the selected samples
            child_entropy = self._entropy(child_y) if self.criterion == 'entropy' else self._gini(child_y) # Calculate child node's entropy using _entropy() or _gini() method
            children_entropy += count / n_samples * child_entropy # Calculate weighted average of child node's entropy
        return parent_entropy - children_entropy # Calculate and return information gain

    def _entropy(self, y):
        # Calculate entropy of a target variable
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-6))

    def _gini(self, y):
        # Calculate Gini index of a target variable
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _entropy_gain(self, X, y, feature):
        # Calculate entropy gain of a feature
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
     if not  isinstance(tree,dict): return tree
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
def chi2_test(X, y): #An often quoted guideline for the validity of this calculation is that the test should be used only if the observed and expected frequencies in each cell are at least 5.
    contingency_table = pd.crosstab(X,y)
    _, pval, _, _ = chi2_contingency(contingency_table)  #pass to chi2 function a contingency table
    return pval


















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

    #post-pruning
    print()
    print("post-pruning by reduce error:")
    outlook_postpruning = np.array([12, 13, 14, 13])
    temperature_postpruning = np.array([4, 15, 4, 4])
    humidity_postpruning = np.array([7, 6, 6, 7])
    wind_postpruning = np.array([9, 9, 9, 8])
    y_postpruning=np.array([11,11,10,10])
    decisiontree = DecisionTree()
    decisiontree.fit(X, y)
    Xval = np.vstack((outlook_postpruning, temperature_postpruning, humidity_postpruning, wind_postpruning)).T
    print("the tree before the pruning:")
    view_tree(decisiontree.get_tree(),embedded_values=value_encoding)
    decisiontree.reduce_error_pruning(Xval)
    print()
    print("the tree after the pruning")
    view_tree(decisiontree.get_tree(), embedded_values=value_encoding)

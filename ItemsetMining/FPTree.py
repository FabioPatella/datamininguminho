import itertools

from FpTreeNode import FPTreeNode
from TransactionDataset import TransactionDataset
from collections import Counter

class FPTree:
    def __init__(self, transactionaldataset, minsup):
        self.item_frequencies = transactionaldataset.get_item_counts()
        self.frequentpatternset = {}
        for item, frequence in self.item_frequencies.items():  # generating the frequent pattern set
            if (frequence >= minsup): self.frequentpatternset[item] = frequence
        self.frequentpatternset = sorted(self.frequentpatternset.items(), key=lambda x: x[1],
                                         reverse=True)  # sorting by decreasing value the frequent pattern set
        self.ordereditemset = []
        print(self.frequentpatternset)
        for transaction in transactionaldataset.get_transactions():  # generating the ordered-item set
            newset = []
            for item in self.frequentpatternset:
                if (item[0] in transaction): newset.append(item[0])
            self.ordereditemset.append(newset)
        print(self.ordereditemset)
        self.build_tree()
        sorted_ordereditemset = sorted(self.frequentpatternset, key=lambda x: x[1])
        items = [x[0] for x in sorted_ordereditemset]
        print(items)
        self.generate_frequent_patterns(items)

    def generate_frequent_patterns(self, items):
        self.frequentpattern = {}
        for item in items:
            self.frequentpattern[item]=[]
        for item in items:
            for node in self.rootnode.get_nexts():
                self.search_item(item,node,[])

        print(self.frequentpattern)
    def generate_conditional_frequent_patterns(self):
        self.conditionalfrequentpattern={}
        #for item,patterns in self.frequentpattern:



    def search_item(self, item, currentnode, pattern):
        if(currentnode.get_item()==item):
           if(pattern!=[]):  self.frequentpattern[item].append((pattern,currentnode.get_frequency()))
        else:
            pattern.append(currentnode.get_item())
            for node in currentnode.get_nexts():
                self.search_item(item,node,pattern[:])




    def build_tree(self):
        self.rootnode = FPTreeNode('root')
        for itemset in self.ordereditemset: self.update_tree(itemset, self.rootnode)

    def update_tree(self, itemset, startingnode):
        foundnode = False
        for node in startingnode.get_nexts():
            if node.item == itemset[0]:
                foundnode = True
                node.updatefrequency()
                itemset.pop(0)  # removing current item
                if len(itemset) > 0:
                    return self.update_tree(itemset, node)
                else:
                    return
        if not foundnode:
            newnode = FPTreeNode(itemset[0])
            startingnode.addNext(newnode)
            itemset.pop(0)
            if len(itemset) > 0:
                return self.update_tree(itemset, newnode)
            else:
                return

    def view_tree(self, currentnode):
        for node in currentnode.get_nexts():
            print(node.get_item() + "," + str(node.get_frequency()), end="")
            if len(node.get_nexts()) > 0:
                print(":[ ", end="")
                self.view_tree(node)
                print(" ]", end="")

    def getroot(self):
        return self.rootnode






if __name__ == '__main__':
    transactions = [
        ['E', 'K', 'M', 'N', 'O', 'Y'],
        ['D', 'E', 'K', 'N', 'O', 'Y'],
        ['A', 'E', 'K', 'M'],
        ['C', 'K', 'M', 'U', 'Y'],
        ['C', 'E', 'I', 'K', 'O', 'O']
    ]

    # create a TransactionDataset object
    dataset = TransactionDataset(transactions)
    fptree = FPTree(dataset, 3)
    fptree.view_tree(fptree.getroot())



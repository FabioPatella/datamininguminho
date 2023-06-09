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

        for transaction in transactionaldataset.get_transactions():  # generating the ordered-item set
            newset = []
            for item in self.frequentpatternset:
                if (item[0] in transaction): newset.append(item[0])
            self.ordereditemset.append(newset)
        self.build_tree()
        sorted_ordereditemset = sorted(self.frequentpatternset, key=lambda x: x[1])
        items = [x[0] for x in sorted_ordereditemset]
        self.generate_frequent_patterns(items)

    def generate_frequent_patterns(self, items):
        self.frequentpattern = {}
        for item in items:
            self.frequentpattern[item]=[]
        for item in items:
            for node in self.rootnode.get_nexts():
                self.search_item(item,node,[])


        self.generate_conditional_frequent_patterns()
    def generate_conditional_frequent_patterns(self):
        """
        get conditional frequent patterns from conditional pattern base
        """
        self.conditionalfrequentpattern={}
        for item in self.frequentpattern.keys():
            self.conditionalfrequentpattern[item]={}
            for pattern in self.frequentpattern[item]: #iterate over all possible patterns
                patternsitem=pattern[0]
                for permutation in itertools.permutations(patternsitem): # iterate over all possible permutations of the pattern
                    currentpatterns=self.conditionalfrequentpattern[item]
                    sortedpermutation=sorted(permutation) #order the permutation to avoi considering the same sub pattern twice
                    if sortedpermutation not in list(currentpatterns.keys()):
                        self.conditionalfrequentpattern[item][tuple(sortedpermutation)] = 0
                        for patterntocheck in self.frequentpattern[item]: #check that the current permutation is in other patterns and if in that case update the frequency
                            patternsitemtocheck=patterntocheck[0]
                            frequency=patterntocheck[1]
                            if all(item2 in patternsitemtocheck for item2 in sortedpermutation):
                                self.conditionalfrequentpattern[item][tuple(sortedpermutation)]=self.conditionalfrequentpattern[item][tuple(sortedpermutation)] + frequency
        maxcondpattern={}

        for item in self.conditionalfrequentpattern.keys(): # takes the pattern with the greatest frequency

            if(len(self.conditionalfrequentpattern[item]))>0:
             bestpattern = max(self.conditionalfrequentpattern[item], key=lambda k: self.conditionalfrequentpattern[item][k])
             frequency=self.conditionalfrequentpattern[item][bestpattern]
             maxcondpattern[item]=(bestpattern,frequency)

        self.conditionalfrequentpattern=maxcondpattern
        self.generate_final_frequent_patterns()

    def generate_final_frequent_patterns(self):
        """
         generate frequent patterns from conditional pattern base
        """
        self.finalpatterns={}
        for item in self.conditionalfrequentpattern.keys():
            self.finalpatterns[item]=[]
            pattern=self.conditionalfrequentpattern[item][0]
            frequency=self.conditionalfrequentpattern[item][1]
            patternlist= list(pattern)
            combinations = []

            for n in range(1, len(patternlist) + 1):
                combinations_n = list(itertools.combinations(patternlist, n))
                combinations.extend(combinations_n)
            for patterncomb in combinations:
                patterncomb=list(patterncomb)
                patterncomb.append(item)
                self.finalpatterns[item].append((patterncomb,frequency))











    def search_item(self, item, currentnode, pattern):
        """
        recursive method useful to build conditional patterns
        :param item: item to search for
        :param currentnode: current node to start
        :param pattern: the pattern already visited , it start with a empty list
        :return:
        """
        if(currentnode.get_item()==item):
           if(pattern!=[]):  self.frequentpattern[item].append((pattern,currentnode.get_frequency()))
        else:
            pattern.append(currentnode.get_item())
            for node in currentnode.get_nexts():
                self.search_item(item,node,pattern[:])




    def build_tree(self):
        """
         build the tree for a collection of itemsets
        """
        self.rootnode = FPTreeNode('root')
        for itemset in self.ordereditemset: self.update_tree(itemset, self.rootnode)

    def update_tree(self, itemset, startingnode):
        """
        recursive method to build the tree
        :param itemset: itemset to build the tree
        :param startingnode: current node explored
        :return:
        """
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
        """
        print a visual representation of the fptree
        :param currentnode: ndoe from which to start printing the tree
        """
        for node in currentnode.get_nexts():
            print(node.get_item() + "," + str(node.get_frequency()), end="")
            if len(node.get_nexts()) > 0:
                print(":[ ", end="")
                self.view_tree(node)
                print(" ]", end="")

    def getroot(self):
        return self.rootnode
    def get_frequent_patterns(self):
        return self.finalpatterns








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
    print("fptree:")
    fptree.view_tree(fptree.getroot())
    print()
    print("frequent patterns:")
    print(fptree.get_frequent_patterns())





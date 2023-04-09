import itertools

from AprioriAlgorithm import AprioriAlgorithm
from TransactionDataset import TransactionDataset


class AssociationRules:
    def __init__(self, frequent_itemsets, transaction_dataset):
        self.frequent_itemsets = frequent_itemsets
        self.transaction_dataset = transaction_dataset
        self.rules = []
    def generate_rule(self):
        for firstitemset in self.frequent_itemsets:
            for seconditemset in self.frequent_itemsets:
                if(self.check_association(firstitemset,seconditemset)):
                    firstsupport=self.transaction_dataset.get_support(firstitemset)
                    secondsupport = self.transaction_dataset.get_support(seconditemset)
                    finalsupport=self.transaction_dataset.get_support(list(set(firstitemset + seconditemset)))
                    confidence=finalsupport/firstsupport
                    self.rules.append(((firstitemset,seconditemset),finalsupport,confidence))

    def print_rules(self):
      for rule in self.rules:
          items=rule[0]
          firstitem=items[0]
          seconditem=items[1]
          support=rule[1]
          confidence=rule[2]
          print(f"{firstitem} => {seconditem} support={support} confidence={confidence}")





    def check_association(self,firstitem,seconditem): #return true if the association can be inserted in the rules
        if any(element in firstitem for element in seconditem): return False
        if sorted(firstitem)==sorted(seconditem):return False
        return True




if __name__ == '__main__':
    # create a TransactionDataset object from a list of transactions
    transactions = [[1,3,5],[1,3,5], [1,5],[1,5],[1,3]]
    transaction_dataset = TransactionDataset(transactions)

    frequent_items=[[1],[3],[5],[1,3],[1,5],[3,5],[1,3,5]]

    association_rules = AssociationRules(frequent_items, transaction_dataset)
    association_rules.generate_rule()
    association_rules.print_rules()
    # print the association rules



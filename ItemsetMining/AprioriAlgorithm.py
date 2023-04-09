import itertools

from TransactionDataset import TransactionDataset


class AprioriAlgorithm:
    def __init__(self, transaction_dataset, minsup):
        self.transaction_dataset = transaction_dataset
        self.minsup = minsup
        self.frequent_itemsets = []

    def generate_frequent_itemsets(self):
        # get frequent items
        frequent_items = self.transaction_dataset.get_frequent_items(self.minsup)

        # initialize the list of current frequent itemsets
        current_frequent_itemsets = [[item] for item in frequent_items]

        # iterate until there are no more frequent itemsets
        while current_frequent_itemsets:
            # add the current frequent itemsets to the list of all frequent itemsets
            self.frequent_itemsets += current_frequent_itemsets

            # generate the candidate itemsets from the current frequent itemsets
            candidate_itemsets = self.generate_candidate_itemsets(current_frequent_itemsets)

            # compute the support of each candidate itemset and filter out infrequent itemsets
            frequent_itemsets = self.filter_infrequent_itemsets(candidate_itemsets)

            # update the list of current frequent itemsets
            current_frequent_itemsets = frequent_itemsets

        # sort the frequent itemsets by length and then lexicographically
        self.frequent_itemsets.sort(key=lambda itemset: (len(itemset), itemset))

    def generate_candidate_itemsets(self, frequent_itemsets):
        # generate candidate itemsets by joining frequent itemsets with themselves
        candidate_itemsets = []
        for i, itemset1 in enumerate(frequent_itemsets):
            for itemset2 in frequent_itemsets[i + 1:]:
                if itemset1[:-1] == itemset2[:-1]:
                    candidate_itemsets.append(itemset1 + [itemset2[-1]])
        return candidate_itemsets

    def filter_infrequent_itemsets(self, candidate_itemsets):
        # count the support of each candidate itemset
        itemset_counts = {}
        for transaction in self.transaction_dataset.get_transactions_with_frequent_items():
            for itemset in candidate_itemsets:
                if set(itemset).issubset(transaction):
                    if tuple(itemset) in itemset_counts:
                        itemset_counts[tuple(itemset)] += 1
                    else:
                        itemset_counts[tuple(itemset)] = 1

        # filter out infrequent itemsets
        frequent_itemsets = []
        for itemset, count in itemset_counts.items():
            if count >= self.minsup:
                frequent_itemsets.append(list(itemset))

        return frequent_itemsets


class AssociationRules:
    def __init__(self, frequent_itemsets, transaction_dataset, minconf):
        self.frequent_itemsets = frequent_itemsets
        self.transaction_dataset = transaction_dataset
        self.minconf = minconf
        self.rules = []
        self.generate_rules()

    def generate_rules(self):
        for itemset in self.frequent_itemsets:
            if len(itemset) > 1:
                for i in range(1, len(itemset)):
                    antecedents = itertools.combinations(itemset, i)
                    for antecedent in antecedents:
                        consequent = list(set(itemset) - set(antecedent))
                        conf = self.calculate_confidence(antecedent, consequent)
                        if conf >= self.minconf:
                            rule = (antecedent, consequent, conf)
                            self.rules.append(rule)

        # sort the rules by confidence
        self.rules.sort(key=lambda rule: rule[2], reverse=True)




if __name__ == '__main__':
    # define the transaction dataset
    transactions = [[1,3,4,6],[2,3,5],[1,2,3,5],[1,5,6]]
    transaction_dataset = TransactionDataset(transactions)

    # define the minimum support threshold
    minsup = 2

    # create an AprioriAlgorithm object with the transaction dataset and the minimum support threshold
    apriori_algorithm = AprioriAlgorithm(transaction_dataset, minsup)

    # generate the frequent itemsets using the Apriori algorithm
    apriori_algorithm.generate_frequent_itemsets()

    # print the frequent itemsets
    print("the frequent itemsets are")
    for itemsets in apriori_algorithm.frequent_itemsets:
            print(f"{itemsets} count :  {transaction_dataset.get_support(itemsets)}")





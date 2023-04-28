import unittest

from AprioriAlgorithm import AprioriAlgorithm
from AssociationRules import AssociationRules
from TransactionDataset import TransactionDataset


class ItemSetMiningTest(unittest.TestCase):
    def test_dataset(self):
        transactions = [
            ['apple', 'banana', 'cherry'],
            ['apple', 'banana'],
            ['apple', 'pear'],
            ['banana', 'cherry'],
            ['banana', 'pear'],
            ['cherry', 'pear'],
            ['apple', 'cherry', 'pear'],
            ['apple']
        ]
        dataset = TransactionDataset(transactions)

        # compute the frequent items
        minsup = 5
        frequent_items = dataset.get_frequent_items(minsup)
        self.assertEqual(frequent_items,['apple'])

        # get the transactions with frequent items
        frequent_transactions = dataset.get_transactions_with_frequent_items()
        self.assertEqual(frequent_transactions,[['apple'], ['apple'], ['apple'], [], [], [], ['apple'], ['apple']])
    def testapriori(self):
        transactions = [[1, 3, 4, 6], [2, 3, 5], [1, 2, 3, 5], [1, 5, 6]]
        transaction_dataset = TransactionDataset(transactions)

        # define the minimum support threshold
        minsup = 2

        # create an AprioriAlgorithm object with the transaction dataset and the minimum support threshold
        apriori_algorithm = AprioriAlgorithm(transaction_dataset, minsup)
        apriori_algorithm.generate_frequent_itemsets()

        # generate the frequent itemsets using the Apriori algorithm
        self.assertEqual(apriori_algorithm.frequent_itemsets,[[1],[2],[3],[5],[6], [1, 3],[1, 5],
            [1, 6],[3, 2],[3, 5],[5, 2],[3, 5, 2]])


if __name__ == '__main__':
    unittest.main()

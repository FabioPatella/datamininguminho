# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class TransactionDataset:
    def __init__(self, transactions):
        self.transactions = transactions
        self.frequent_items = None

    def get_frequent_items(self, minsup):
        # count the frequency of each item
        item_counts = {}
        for transaction in self.transactions:
            for item in transaction:
                if item in item_counts:
                    item_counts[item] += 1
                else:
                    item_counts[item] = 1

        # filter out infrequent items
        frequent_items = set()
        for item, count in item_counts.items():
            if count >= minsup:
                frequent_items.add(item)

        # store the frequent items in reverse frequency order
        frequent_items = list(frequent_items)
        frequent_items.sort(key=lambda item: item_counts[item], reverse=True)
        self.frequent_items = frequent_items

        # return the frequent items
        return frequent_items

    def get_transactions_with_frequent_items(self):
        # ensure that frequent items have been computed
        if self.frequent_items is None:
            raise ValueError("Frequent items have not been computed yet.")

        # filter out infrequent items from each transaction
        frequent_transactions = []
        for transaction in self.transactions:
            frequent_transaction = [item for item in transaction if item in self.frequent_items]
            frequent_transactions.append(frequent_transaction)

        # return the frequent transactions
        return frequent_transactions

    def get_support(self, itemset):
        # count the frequency of the itemset in the transactions
        count = 0
        for transaction in self.transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        # compute the support as the count of transaction containing the item
        support = count
        return support


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # create a dataset of transactions
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

    # create a TransactionDataset object
    dataset = TransactionDataset(transactions)

    # compute the frequent items
    minsup = 5
    frequent_items = dataset.get_frequent_items(minsup)
    print("Frequent items:", frequent_items)

    # get the transactions with frequent items
    frequent_transactions = dataset.get_transactions_with_frequent_items()
    print("Transactions with frequent items:", frequent_transactions)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

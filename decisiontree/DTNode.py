class DTNode:
    """
    class for node of the decision tree
    """

    def __init__(self, value, previous=None, isleaf=False):
        self.value = value
        self.previous = previous
        self.next = []
        self.isleaf = isleaf

    def get_next(self):
        return self.next

    def get_previous(self):
        return self.previous

    def set_previous(self, previous):
        self.previous = previous

    def set_leaf(self, leaf):
        self.isleaf = leaf

    def add_next(self, node):
        self.next.append(node)

    def get_value(self):
        return self.value

    def isLeaf(self):
        return self.isleaf
    def reset_next(self):
        self.next=[]

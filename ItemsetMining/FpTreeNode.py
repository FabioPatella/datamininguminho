class FPTreeNode:
    def __init__(self,item):
        self.item=item
        self.frequency=1
        self.next=[]
    def updatefrequency(self):
        self.frequency=self.frequency+1
    def addNext(self,node):
        self.next.append(node)
    def get_item(self):
        return self.item
    def get_nexts(self):
        return self.next
    def get_frequency(self):
        return self.frequency

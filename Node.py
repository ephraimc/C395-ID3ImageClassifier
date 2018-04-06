#The Node class forms the structure of a tree
#Label is the name of the attribute such as AU1, AU2...
#Kids is a dictionary that holds two subtrees for each value of the attribute
#leaf is the classification at the leaf, it can either be 1 or 0
#targetCount is the sample size at the leaf
#targetDistribution is the distribution of sample size at the leaf
class Node:
    def __init__(self):
        self.label = None
        self.kids = {}
        self.leaf = None
        self.targetCount = 0
        self.targetProportion = None
    label = None
    kids = None
    leaf = None
    targetCount = None
    targetProportion = None
    print('Classification error on whole data set:' , class_error/classes, '\n')

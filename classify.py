#THIS FILE CONTAINS FUNCTIONS USED TO CLASSIFY A GIVEN EXAMPLE
import scipy.io as sio
import functions as f
import numpy as np

#finds the index of a given attribute in an array, for example AU1 has index 0
def findIndex(attributes, target):
    for i in range(len(attributes)):
        if attributes[i] == target:
            return i

#classifies when given an example and a tree
def classify(tree, examples, attributes):
    temp = []
    if tree.label is None:
        temp.append(tree.leaf)
        temp.append(tree.targetCount)
        temp.append(tree.targetProportion)
        output = np.vstack(temp)
        return output
    else:
        index = findIndex(attributes, tree.label)
        return classify(tree.kids[examples[index]], examples, attributes)

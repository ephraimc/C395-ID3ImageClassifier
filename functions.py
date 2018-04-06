#THIS FILE CONTAINS ALL FUNCTIONS USED FOR TREE MAKING

import numpy as np
import math
import Node as n


#generates an array of attributes labels AU1, AU2, AU3... AU45
def makeAttributes(examples): #generates a row of AU to keep track of attributes
    header = []
    for i in range(len(examples[0])):
        string = 'AU'
        string += str(i+1)
        header.append(string)
    header = np.array(header)
    return header

#removes attribute from the attribute list after it became a root
def nextAttributes(original, target):
    new = np.delete(original, target)
    return new

#selects relevant examples for the subtree of a root
def nextExampleSet(examples, target, value):
    temp = []
    output = None
    count = 0
    for i in range( len(examples) ):
        if examples[i][target] == value:
            count += 1
            row = examples[i:i+1, : ]
            row = np.delete(row, target, 1)
            temp.append(row)
    output = np.vstack(temp)
    return output

#changes target emotion to 1 and the rest to 0
def bin(targets, value):
    temp = []
    binaryTargets = []
    for i in range(len(targets)):
        if targets[i] == value:
            temp.append([1])
        else:
            temp.append([0])
    binaryTargets = np.vstack(temp) #this is to keep the dimension the same as input
    return binaryTargets

#selects relevant binaryTargets for the subtree of a root
def nextBinaryTargets(binaryTargets, examples, target, value):
    temp = []
    output = None
    count = 0
    for i in range( len(examples) ):
        if examples[i][target] == value:
            count += 1
            temp.append(binaryTargets[i:i+1])
    output = np.vstack(temp)
    return output

#checks if all binary targets have the same value
def allSame(binaryTargets):
    return all(x == binaryTargets[0] for x in binaryTargets)

#compares information gains of all attributes to select best one as root
def chooseAttribute(binaryTargets, examples):
    informationGains = [] #list of information gains for all remaining attributes
    for i in range(len(examples[0])):
        informationGains.append(informationGain(binaryTargets,examples,i))
    best = informationGains.index(max(informationGains))
    return best

#returns the majority value of binary targets. called when attributes is empty
def majority(binaryTargets):
    countPos = 0
    countNeg = 0
    for i in range( len(binaryTargets) ):
        if binaryTargets[i] == 1:
            countPos += 1
        else:
            countNeg += 0
    if countPos > countNeg:
        return 1
    elif countNeg > countPos:
        return 0
    else:
        return -1

#returns propotion of positive to negative examples in given binaryTargets
def proportion(binaryTargets):
    countPos = 0
    countNeg = 0
    totalCount = len(binaryTargets)
    for i in range(totalCount):
        if binaryTargets[i] == 1:
            countPos += 1
        else:
            countNeg += 1
    if countPos >= countNeg:
        return float(countPos) * 100 / totalCount
    else:
        return float(countNeg) *100 / totalCount

#returns unique values in a list
def uniqueValues(attributes, column):
    column = [row[column] for row in attributes]
    uniqueList = list( set(column) )
    return uniqueList

#recursively builds a tree
def makeTree( examples, attributes, binaryTargets ):
    root = n.Node()
    if allSame(binaryTargets): #when binary targets converge to the same answer
        root.label = None #label is none means it is a leaf
        root.leaf = binaryTargets[0][0] #leaf is the classification 1:yes,0:no
        root.kids = None
        root.targetCount = np.array([len(binaryTargets)])
        root.targetProportion = np.array([proportion(binaryTargets)])
        return root
    elif len(attributes) == 0: #when binary targets do not converge
        root.label = None
        root.leaf = majority(binaryTargets)
        root.kids = None
        root.targetCount = np.array([len(binaryTargets)])
        root.targetProportion = np.array([proportion(binaryTargets)])
        return root
    else: #when we have not reached leaf yet and needs to build subtree
        best = chooseAttribute(binaryTargets, examples)
        root.label = attributes[best]
        root.leaf = None
        root.targetCount = None
        root.targetProportion = None
        for val in range(2):
            if not attributeHasValue(examples, best, val):
                leaf = n.Node()
                leaf.label =  None
                leaf.leaf = majority(binaryTargets)
                leaf.count = np.array([len(binaryTargets)])
                leaf.targetProportion = np.array([proportion(binaryTargets)])
                return leaf

            subtree = makeTree( nextExampleSet(examples, best, val),
                                nextAttributes(attributes, best),
                                nextBinaryTargets(binaryTargets, examples, best, val))
            root.kids[val] = subtree
        return root

#boolean function to check whether an attribute has a certain value
def attributeHasValue(examples, attribute, val):
    for i in range(len(examples)):
        if examples[i][attribute] == val:
            return True
    return False

#calculates information gain for an attribute
def informationGain(binaryTargets, examples, target):
    entropyOfSet = entropy(binaryTargets)
    entropyOfPartitions = []
    uniqueVals = uniqueValues(examples, target)
    temp = []
    exampleCount = 0
    partitionsTotal = 0
    for i in range( len(uniqueVals) ): #for each unique value of an attirbute
        temp.append([]) #create an list to hold its examples
        for j in range( len(examples) ):
            if examples[j][target] == uniqueVals[i]:
                temp[i].append( binaryTargets[j] ) #examples for this attribute value
        exampleCount += len(temp[i])
    for i in range( len(temp) ):
        entropyOfPartitions.append( (-float(len(temp[i])) / exampleCount) * (entropy(temp[i])) )
    for i in range( len(entropyOfPartitions) ):
        partitionsTotal += entropyOfPartitions[i]
    return entropyOfSet + partitionsTotal

#calculates entropy of a given set of binary targets
def entropy(binaryTargets):
    positiveCount = float(0)
    negativeCount = float(0)
    totalCount = len(binaryTargets)
    for i in range(totalCount):
        if binaryTargets[i] == 1:
            positiveCount += 1
        else:
            negativeCount += 1
    if positiveCount == totalCount or negativeCount == totalCount:
        return 0
    else:
        probabilityP = positiveCount / totalCount
        probabilityN = negativeCount / totalCount
        return -(probabilityP) * math.log(probabilityP, 2) - (probabilityN) * math.log(probabilityN, 2)

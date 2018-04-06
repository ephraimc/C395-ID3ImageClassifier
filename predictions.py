#THIS FILE CONTAINS FUNCTIONS WHICH ARE USED TO GENERATE PREDICTIONS GIVEN TREES
import classify as c
import numpy as np

#generates an array of predictions as specified in the CBC manual
#output has shape (numberofexamples, 1), same as the input
def testTrees(treeSet, testSet, attributes, tieBreaker):
    binaryClassification = []
    sampleSize = []
    distribution = []
    for i in range(len(testSet)):
        row = []
        for j in range(len(treeSet)):
            row.append(c.classify(treeSet[j], testSet[i], attributes)[0][0])
        binaryClassification.append(row)
    binaryClassification = np.vstack(binaryClassification)

    for i in range(len(testSet)):
        row = []
        for j in range(len(treeSet)):
            row.append(c.classify(treeSet[j], testSet[i], attributes)[1][0])
        sampleSize.append(row)
    sampleSize = np.vstack(sampleSize)

    for i in range(len(testSet)):
        row = []
        for j in range(len(treeSet)):
            row.append(c.classify(treeSet[j], testSet[i], attributes)[2][0])
        distribution.append(row)
    distribution = np.vstack(distribution)

    output = finalPredictions(binaryClassification, sampleSize, distribution, tieBreaker)
    return output

#finalPredictions generates final predictions by combining binary predictions from 6 trees
#output is an array of a number from 1-6, each representing the predicted emotion
def finalPredictions(binaryPredictions, sampleSize, sampleDistribution, tieBreaker):
    output = []
    for i in range(len(binaryPredictions)):
        positives = []
        for j in range(len(binaryPredictions[i])):
            if binaryPredictions[i][j] == 1:
                positives.append(j)
        if len(positives) == 1:
            output.append(positives[0] + 1)
        elif len(positives) > 1:
            if tieBreaker == 1: #method 1 of tie breaking
                output.append(tieBreakerBySize(positives, sampleSize[i]) + 1)
            if tieBreaker == 2: #method 2 of tie breaking
                output.append(tieBreakerByDistribution(positives, sampleSize[i]) + 1)
        else:
            output.append(findMin(sampleSize[i]) + 1)
    output = np.vstack(output)
    return output

#returns the index containing the largest number in an array
def findMax(array):
    max = array[0]
    maxIndex = 0
    for i in range(len(array)):
        if array[i] > max:
            max = array[i]
            maxIndex = i
    return maxIndex

#returns the index containing the smallest number in an array
def findMin(array):
    min = array[0]
    minIndex = 0
    for i in range(len(array)):
        if array[i] < min:
            min = array[i]
            minIndex = i
    return minIndex

#counts frequency of examples at the leaf and returns the classification with the
#highest frequency of examples
def tieBreakerBySize(positives, sampleSize):
    temp = []
    for i in range(len(positives)):
        temp.append(sampleSize[positives[i]])
    return findMax(sampleSize)

#counts distribution of examples at the leaf and returns the classification
#supported by the strongest majority
def tieBreakerByDistribution(positives, sampleDistribution):
    temp = []
    for i in range(len(positives)):
        temp.append(sampleDistribution[positives[i]])
    return findMax(sampleDistribution)

#when all trees classify an example as negative, weakestNegative chooses
#the tree with the weakest (smallest) sample size at the leaf
def weakestNegative(sampleSize):
    return findMin(sampleSize)

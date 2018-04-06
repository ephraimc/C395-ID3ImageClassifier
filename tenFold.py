#THIS FILE PRODUCES CONFUSION MATRICES AFTER 10-FOLD CROSS VALIDATION
import functions as f
import numpy as np
import classify as c
import evaluations as e
import predictions as p
import random

anger = 1
disgust = 2
fear = 3
happiness = 4
sadness = 5
surprise = 6

#carries out 10 fold cross validation combines results to generate average
#performance measures
def tenFold(examples, attributes, binaryTargets, tieBreaker):
    #randomArray is used to pick out random examples for exach fold
    randomArray = np.array(list(range(0, len(examples))))
    #binaryTargets are reshuffled so it matches the reshuffled example set
    binaryTargetsReshuffled = []
    predictions = None

    #loops through each fold
    for i in range(10):
        trainingExamples = [] #training examples for this fold
        binaryTargetsTraining = [] #training targets for this fold
        testExamples = [] #testing examples for this fold
        rowsToDelete = [] #removes example once it has been selected for testing

        examplesCopy = examples
        targetsCopy = binaryTargets
        np.random.shuffle(randomArray) #shuffles array so we select random examples

        if i < 9:
            bound = 100
        if i == 9: #the 10th fold includes slightly more than 100 examples
            bound = len(examples) - 900
        for j in range(bound):
            #builds testing set by randomly selecting examples
            index = randomArray[0]
            randomArray = np.delete(randomArray, (0), 0)
            testExamples.append(examples[index])
            binaryTargetsReshuffled.append(binaryTargets[index])
            rowsToDelete.append(index)

        #removes examples and targets chosen for testing from training set
        trainingExamples = np.delete(examplesCopy, np.array(rowsToDelete), 0)
        binaryTargetsTraining = np.delete(binaryTargets, np.array(rowsToDelete), 0)
        testExamples = np.vstack(testExamples)


        #BUILDS 6 TREES
        angerTree = f.makeTree(trainingExamples, attributes, f.bin(binaryTargetsTraining, anger))
        disgustTree = f.makeTree(trainingExamples, attributes, f.bin(binaryTargetsTraining, disgust))
        fearTree = f.makeTree(trainingExamples, attributes, f.bin(binaryTargetsTraining, fear))
        happyTree = f.makeTree(trainingExamples, attributes, f.bin(binaryTargetsTraining, happiness))
        sadTree = f.makeTree(trainingExamples, attributes, f.bin(binaryTargetsTraining, sadness))
        surpriseTree = f.makeTree(trainingExamples, attributes, f.bin(binaryTargetsTraining, surprise))

        #PUT 6 TREES FOR THIS FOLD AND PUT INTO AN ARRAY
        treesFold = np.array([angerTree, disgustTree, fearTree, happyTree, sadTree, surpriseTree])

        #CONCATENATE PREDICTIONS FROM EACH FOLD
        foldPredictions = p.testTrees(treesFold, testExamples, attributes, tieBreaker)
        if i == 0:
            predictions = foldPredictions
        if i > 0:
            predictions = np.concatenate((predictions, foldPredictions))


    #Creating the (un-normalized) matrix
    matrix = e.confusion_matrix(predictions, binaryTargetsReshuffled)
    #Running the evaluations of each class
    classes = 6

    displayMatrix(classes, matrix)


    #Running the evaluations of each class for the normalized matrix
    print("\n Evaluations on the NORMALIZED matrix \n")
    #Creating the normalized matrix
    norm_matrix = e.normalize_matrix(matrix)
    displayMatrix(classes, norm_matrix)



#prints out matrix
def displayMatrix(classes, matrix):
    for c in range(1, classes+1):
        print("For class", c ,":" , '\n')
        rates = e.rates_calc(matrix,c)
        recall_sum = rates[0]
        class_rate = rates[2]
        class_error = rates[3]
        e.f1_measure(rates,c)
        print('\n')

    print("The Unweighted Average Recall (UAR) is: ", recall_sum/classes, '\n')
    print('Classification error on whole data set:' , class_error/classes, '\n')
    print('Classification accuracy on whole set:', class_rate/classes, '\n')

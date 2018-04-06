#THIS FILE CONTAINS FUNCTIONS USED TO PRODUCE CONFUSION MATRICES
import numpy as np
import scipy.io as sio

#Function that computes the confusion matrix given
#two column vectors of predicted
#and actual data
def confusion_matrix(predicted, actual):
    data_col = 0
    classes = 6
    matrix = np.zeros(shape=(classes,classes))
    no_examples = len(actual)

    #Loops through total examples, checks if data matches
    for index in range(0,no_examples):
        predicted_class = predicted[index][data_col]
        actual_class = actual[index][data_col]
        if predicted_class==actual_class:
            matrix[actual_class-1][actual_class-1] +=1
        else:
            matrix[actual_class-1][predicted_class-1]+=1


    print("Confusion matrix is: \n ")
    print(matrix, '\n')
    return matrix

#Calculates the total number of examples present for a given class
def total_examples(matrix, target_class):
    total = 0

    for col in range(0, 6):
        total += matrix[target_class-1][col]

    return total

#Calculates recall and precision rates for a target class
def rates_calc(matrix, target_class):
    print("There are ", total_examples(matrix, target_class), " examples for this class.")

    index = target_class-1
    TP = matrix[index][index]
    FN = 0
    FP = 0
    TN = 0

    for col in range(0,6):
        if col!=index:
            FN+=matrix[index][col]

    for row in range(0,6):
        if row!=index:
            FP+= matrix[row][index]

    for row in range(0,6):
        for col in range(0,6):
            if row!=index and col!=index:
                TN+= matrix[row][col]

    if(TP+FN)==0:
        print("Denominator (TP + FN) is 0 for this class" , target_class, "hence recall rate is 0.")
        recall_rate = 0
    else:
        recall_rate = (TP/(TP + FN))*100

    if(TP+FP)==0:
        print("Denominator (TP + FP) is 0 for class " , target_class , "hence precision rate is 0.")
        precision_rate = 0
    else:
        precision_rate = (TP/(TP+FP))*100

    #Calssification accuracy and errors per class
    classification_rate = (TP+TN)/(TP+TN+FP+FN)
    classification_error = 1-classification_rate


    print("Recall rate is: ", recall_rate)
    print("Precision rate is: ", precision_rate)
    print("The classification rate/accuracy is: ", classification_rate)
    print("The classification error is: ", classification_error)
    return recall_rate, precision_rate, classification_rate, classification_error

#Calculates f1 measure for a given class
def f1_measure(rates, target_class):
    recall_rate = rates[0]
    precision_rate = rates[1]
    n = (precision_rate*recall_rate)
    d = (precision_rate+recall_rate)
    if d==0:
        print("Denominator is 0 in f1-measure for class", target_class, "hence f1 measure is 0.")
        return 0
    else:
        f1 = 2*(n/d)
        print("f1-measure: ", f1)

    return f1

#normalize matrices for when input data is biased
def normalize_matrix(matrix):
    classes = 6

    norm_matrix = np.zeros(shape=(classes,classes))

    for i in range(0,6):
        sum_row = total_examples(matrix,i+1)
        for j in range(0,6):
            norm_matrix[i][j] = matrix[i][j]/sum_row

    print("NORMALIZED CONFUSION MATRIX: \n ", norm_matrix)
    return norm_matrix

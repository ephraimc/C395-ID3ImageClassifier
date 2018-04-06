import tenFold as t
import scipy.io as sio
import makeFinalTrees as m
import functions as f
import predictions as p
import random
import numpy as np
import config
import pickle

# TO BE CONFIGURED BY USERS IN CONFIG.PY
TrainingFile = config.TrainingFile
TestFile = config.TestFile
tieBreaker = config.tieBreaker

#LOAD TRAINING DATA
data = sio.loadmat(TrainingFile)
examples = data['x']
binaryTargets = data['y']
attributes = f.makeAttributes(examples)

#LOAD TESTING DATA
data2 = sio.loadmat(TestFile)
examples2 = data2['x']


print("-------------------BEGIN PRODUCING CONFUSION MATRICES--------------------")
print("CONFUSION MATRICES FOR TRAINING FILE %s USING METHOD %d ARE AS FOLLOWS:" % (TrainingFile, tieBreaker))
#PRODUCE CONFUSION MATRICES ON TRAINING FILE
t.tenFold(examples, attributes, binaryTargets, tieBreaker)
print("")
print("")

print("-------------------------BEGIN MAKING TREES------------------------------")
#CREATE FINAL TREES BASED ON TRAINING FILE AND PRODUCE PREDICTIONS GIVEN TEST FILE
treeSet = m.makeFinalTrees(examples, attributes, binaryTargets)
filehandler = open("treeSet.pkl", 'w')
pickle.dump(treeSet,filehandler)
print("---------------TREES NOW SAVED IN treeSet.pkl file-----------------------")
print("")
print("")
predictions = p.testTrees(treeSet, examples2, attributes, tieBreaker)
print("------PREDICTIONS BY OUR TREES ON YOUR DATA IN %s ARE AS FOLLOWS-----" % TestFile)
print(predictions)

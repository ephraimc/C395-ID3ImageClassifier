#THIS FILE CONTAINS THE FUNCTIONS TO TRAIN TREES ON ENTIRE DATA SET FOR SUBMISSION
import functions as f
import numpy as np
anger = 1
disgust = 2
fear = 3
happiness = 4
sadness = 5
surprise = 6

def makeFinalTrees(examples, attributes, binaryTargets):
    binaryTargets_1 = f.bin(binaryTargets , anger)
    binaryTargets_2 = f.bin(binaryTargets, disgust)
    binaryTargets_3 = f.bin(binaryTargets, fear)
    binaryTargets_4 = f.bin(binaryTargets, happiness)
    binaryTargets_5 = f.bin(binaryTargets, sadness)
    binaryTargets_6 = f.bin(binaryTargets, surprise)

    tree_list = []
    angerTree = f.makeTree(examples, attributes, binaryTargets_1)

    disgustTree = f.makeTree(examples, attributes, binaryTargets_2)

    fearTree = f.makeTree(examples, attributes, binaryTargets_3)

    happyTree = f.makeTree(examples, attributes, binaryTargets_4)

    sadnessTree = f.makeTree(examples, attributes, binaryTargets_5)

    surpriseTree = f.makeTree(examples, attributes, binaryTargets_6)

    treesList = np.array([angerTree, disgustTree, fearTree, happyTree, sadnessTree, surpriseTree])
  
    return treesList

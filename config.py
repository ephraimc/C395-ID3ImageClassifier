# TRAINING FILE IS USED TO PRODUCE CONFUSION MATRICES AND TRAING FINAL TREES
TrainingFile = "cleandata_students.mat"
# TEST FILE IS FILE OF YOUR CHOICE TO TEST OUR TREES
TestFile = "noisydata_students.mat"
#SELECT TIEBREAKER METHOD 1 or 2 WHEN THERE IS AMBIGUITY
#method 1 is tiebreak by sample sample size
#method 2 is tiebreka by sample distribution
tieBreaker = 1

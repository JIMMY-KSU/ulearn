#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import os
_here       = os.path.dirname(os.path.realpath(__file__))
from time import time
sys.path.append("../tools/")
from sklearn.svm import SVC
from email_preprocess import preprocess


### features_train and features_test are the features for the training
# ### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
t0 = time()
for C in [10000]:
    clf = SVC(kernel='rbf',C=C)

    # Take just 1% ish of the data to train with for speed sake
    clf.fit(features_train[:len(features_train)//50], labels_train[:len(labels_train)//50])
    print(f"Linear SVC took {round(time()-t0,3)} seconds to fit")

    accuracy = clf.score(features_test, labels_test)
    print(f"Accuracy was: {accuracy}, C={C}")

    # print(f"Prediction for test feature 10: {clf.predict(features_test[10])}")
    # print(f"Prediction for test feature 26: {clf.predict(features_test[26])}")
    #
    # print(f"Prediction for test feature 50: {clf.predict(features_test[50])}")
    print(f"Chris (1) was predicted {sum(clf.predict(features_test))} times")

#########################################################



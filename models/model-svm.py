################################################################################
# model-svm.py
#
# This file comprises the SVM model which takes in numpy data files and fits an
# SVM to the data and does some testing
################################################################################

from __future__ import division
from numpy import genfromtxt
import numpy as np
import os, sys, time
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing, svm, grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from scipy.misc import imresize
sys.dont_write_bytecode = True
from loading_methods import load_data, windows, resize
from preprocessing_methods import zero_meaning, normaliser, standardiser
import matplotlib.pyplot as plt

#   Usage:
#
#     model.py -i <str>
#
#   where
#     --input <str> this is the directory in which input data files are stored
#     -i
#
#     --dataType <str> this is the string to define datatype: "windows" or "radargrams"
#     -d
#

parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('-i','--input', dest='dataDir', metavar='dataDir', type=str,
                   help='string for data directory', required=True)

parser.add_argument('-d','--datatype', dest='dataType', metavar='dataType', type=str,
                   help='string for data type', required=True)

args = parser.parse_args()

dataDir = str(args.dataDir)
dataType = str(args.dataType)

#######################################################################
# 1: load up data, resize if not windows.
#######################################################################

if dataType=="windows":
    new_size = 0
else:
    new_size = (32, 32)

print "Loading data ..."
if os.path.isfile(dataDir+"/X_train.npy"):
    X_train = np.load(dataDir+"/X_train.npy")
    Y_train = np.load(dataDir+"/Y_train.npy")
    X_test = np.load(dataDir+"/X_test.npy")
    Y_test = np.load(dataDir+"/Y_test.npy")
    Y_train = Y_train.T
    Y_test = Y_test.T
else:
    X_train, Y_train = load_data("train", dataDir, dataType, new_size)
    X_test, Y_test = load_data("test", dataDir, dataType, new_size)

######################################################
# 2: preprocessing
######################################################
# background removal using eigenvalues
# histogram eqaulisation
# normalisation

print "Mean subtraction ..."

X_train, X_test, mean = zero_meaning(X_train, X_test)

X_train, val1, val2 = normaliser(X_train, "train")
X_test, val1, val2 = normaliser(X_test, "test", val1, val2)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_train = Y_train.astype(np.uint8)
Y_test = Y_test.astype(np.uint8)
print "Resizing ..."

new_size = (32,32)
X_train = resize(X_train, new_size)
X_test = resize(X_test, new_size)

def plot_example():
    print "plotting example ..."
    clean_idx = np.where(Y_train==0)
    mine_idx = np.where(Y_train==1)

    conv = np.hstack((X_train[:,:,clean_idx[0][i]],X_train[:,:,mine_idx[0][i]]))
    plt.imshow(conv, cmap='gray')
    plt.show()

# plot_example()

print "Reshape for SVM ..."

X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
X_train = X_train.T

X_test = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1],X_test.shape[2]))
X_test = X_test.T

print "Training X shape : "+str(X_train.shape)
print "Training Y shape : "+str(Y_train.shape)
print "Test X shape     : "+str(X_test.shape)
print "Test Y shape     : "+str(Y_test.shape)

print ""

print "Number of radargrams with a mine : "
print "     - Train : "+str(len(Y_train[Y_train==1])) +"/"+str(len(Y_train))
print "     - Test  : "+str(len(Y_test[Y_test==1])) +"/"+str(len(Y_test))
print ""

######################################################
# 4: pass windows through Neural Network
######################################################

print "Build model ..."

# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(Y_train, n_iter=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
# grid.fit(X_train, Y_train)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

clf = svm.SVC(kernel='rbf', gamma=1, C=1, probability=True)
# clf = svm.SVC(probability=True)

clf.fit(X_train, Y_train)

Y_predict = clf.predict(X_test)

print confusion_matrix(Y_test, Y_predict)


#

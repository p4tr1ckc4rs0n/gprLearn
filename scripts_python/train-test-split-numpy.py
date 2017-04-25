#!/usr/bin/python

################################################################################
# train-test-split-numpy.py
#
# This python script splits the training and test data numpy files
################################################################################

from sklearn.cross_validation import train_test_split
import os
import numpy as np
import shutil
import argparse
import matplotlib.pyplot as plt

#   Usage:
#
#     train-test-split.py -i <str>
#
#   where
#     --input <str> this is the directory in which the .csv files are stored
#     -i
#
#     --output <str> this is the directory in which numpy files are saved
#     -o

parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('-i','--input', dest='dataDir', metavar='dataDir', type=str,
                   help='string for data directory', required=True)

parser.add_argument('-o','--output', dest='outputDir', metavar='outputDir', type=str,
                   help='string for output data directory', required=True)

args = parser.parse_args()

# switch to correct directory
dataDir = str(args.dataDir)
outputDir = str(args.outputDir)
os.chdir(dataDir)

# load the file with "X_data" or "Y_data" in the name
for file in os.listdir(dataDir):
    if "X_data" in file:
        X = np.load("X_data.npy")

    if "Y_data" in file:
        Y = np.load("Y_data.npy")

# split in to train and test sets
idx = range(X.shape[2])
idx_train, idx_test = train_test_split(idx, test_size=0.25)

# assign data to variables
X_train = X[:,:,idx_train]
X_test = X[:,:,idx_test]

Y_train = Y[idx_train]
Y_test = Y[idx_test]

# save to outputDir
os.chdir(outputDir)

np.save("X_train",X_train)
np.save("Y_train",Y_train)
np.save("X_test",X_test)
np.save("Y_test",Y_test)


#

#!/usr/bin/python

################################################################################
# train-test-split-files.py
#
# This python script splits the training and test png files in to two
# separate numpy files
################################################################################

from __future__ import division
from sklearn.cross_validation import train_test_split
import os, sys, gc
import numpy as np
import shutil
import argparse
import matplotlib.pyplot as plt
import cv2

#   Usage:
#
#     train-test-split.py -i <str>
#
#   where
#     --input <str> this is the directory in which the .csv files are stored
#     -i
#
#     --output <str> this is the directory in which train/test split files are saved
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

imagelist = []

# create list of filenames
for file in os.listdir(dataDir):
    if "with" in file and file.endswith(".png"):
        imagelist.append(file)

##  Split radargram and mine .png lists in to train/test directories
train_list, test_list = \
    train_test_split(imagelist, test_size=0.25)

def save_data(data_list, data_type):
    print data_type+" progress : "
    count = 0
    for file_name in data_list:
        # show progress
        sys.stdout.write("\r%d%%" % np.round((count/len(data_list))*100))
        sys.stdout.flush()

        # loop through list and save images in to numpy array
        im = cv2.imread(dataDir+"/"+file_name,0)

        # resize now to save memory
        im = cv2.resize(im, (64,64))
        # cv2.imshow("image",im)

        if count == 0:
            # preallocate size of array
            X = np.empty([im.shape[0],im.shape[1],len(data_list)])
            Y = np.empty(len(data_list))

        X[:,:,count] = im
        if "without" in file_name:
            Y[count] = 0
        else:
            Y[count] = 1

        count += 1

    return X, Y

# now create list of files and classes
X_train, Y_train = save_data(train_list, "Train")
print ""
X_test, Y_test = save_data(test_list, "Test")

# save to outputDir
os.chdir(outputDir)

np.save("X_train",X_train)
np.save("Y_train",Y_train)
np.save("X_test",X_test)
np.save("Y_test",Y_test)

print ""


#

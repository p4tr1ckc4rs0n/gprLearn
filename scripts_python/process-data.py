#!/usr/bin/python

################################################################################
# process-data.py
#
# A simple script to load up the data and save it to png format in the correct
# way
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
import sys, os
import argparse
from os.path import expanduser
import Image
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import gc
sys.dont_write_bytecode = True
from PIL import Image
import cv2

#   Usage:
#
#     process-data.py -i <str> -o <str>
#
#   where
#     --input <str> this is the directory in which input data files are stored
#     -i
#
#     --output <str> this is the directory the images will end up in
#     -o
#
#     --mineloc <str> this is the directory where positions of mines are stored
#     -m
#
#     --fileid <itr> optional input if the file id must be specified. Can only specify
#     -f             file id if one file is being changed
#

# parse input commands
parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('-i','--input', dest='dataDir', metavar='dataDir', type=str,
                   help='string for data directory', required=True)

parser.add_argument('-o','--output', dest='outputDir', metavar='outputDir', type=str,
                   help='string for output data directory', required=True)

parser.add_argument('-m','--mineDir', dest='mineDir', metavar='mineDir', type=str,
                   help='string to define directory where mineposition files are stored', required=True)

args = parser.parse_args()

# switch to correct directory
dataDir = str(args.dataDir)
outputDir = str(args.outputDir)
mineDir = str(args.mineDir)

os.chdir(outputDir)

#######################################################################
# Compile all relevant data in to folders with correct nomenclature
#######################################################################

# initialise vairables

count = 0

# fill list with all .csv files
filelist = [ f for f in listdir(dataDir) if isfile(join(dataDir,f)) ]
cutoff = 120
mindataset = 0
maxdata = 0

# get absolute max and min of dataset
for file in natsorted(filelist):
    if file.endswith(".csv"):
        data = np.loadtxt(open(dataDir+"/"+file,"rb"),delimiter=",")
        data = data[cutoff:data.shape[0],:]

        datarange = np.amax(data) - np.amin(data)

        if abs(datarange) < 2000:
            if count == 0:
                maxdataset = np.amax(data)
                mindataset = np.amin(data)
            else:
                maxdata = np.amax(data)
                mindata = np.amin(data)

                if maxdata > maxdataset:
                    maxdataset = maxdata

                if mindata < mindataset:
                    mindataset = mindata

        count += 1

count = 0

for file in natsorted(filelist):
    if file.endswith(".csv"):
        print "Loading data from file: "+file

        image_name = file.replace(".csv","")

        if "without" in image_name:
            fileid = image_name.replace("without","")
        else:
            fileid = image_name.replace("with","")

        # load csv data, cut off band and relative rescale
        data = np.loadtxt(open(dataDir+"/"+file,"rb"),delimiter=",")
        data = data[cutoff:data.shape[0],:]

        rescaled = np.divide((data - mindataset),(maxdataset - mindataset))*255
        # rescaled = np.divide((data - np.amin(data)),(np.amax(data) - np.amin(data)))*255

        np.savetxt(image_name+".csv", data, delimiter=",")

        # save image
        # im = cv2.imread(rescaled,0)
        cv2.imwrite(image_name+".png",rescaled)

        # save mine file and image
        if "without" not in file:
            # mine data file
            minedata = np.loadtxt(open(mineDir+"/"+file.replace(".csv","")+"_minepos.csv","rb"),delimiter=",")
            np.savetxt("mine"+fileid+".csv", minedata, delimiter=",")

        count += 1
        plt.clf()

print "All data files saved"
print "----------------------------------------------------"
print ""

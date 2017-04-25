#!/usr/bin/python

################################################################################
# remove-skewed-data.py
#
# A simple script to remove erroneous data produced by the OLD gprMax software.
# WARNING: The method used to identify erroneous data is dubious and not very
# scientific.
################################################################################

import numpy as np
import sys, os
from loading_methods import load_data
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import argparse
sys.dont_write_bytecode = True

#   Usage:
#
#     process-data.py -i <str> -o <str>
#
#   where
#     --input <str> this is the directory in which input data files are stored
#     -i
#

# parse input commands
parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('-i','--dataDir', dest='dataDir', metavar='dataDir', type=str,
                   help='string to define directory where mineposition files are stored', required=True)

args = parser.parse_args()

# switch to correct directory
dataDir = str(args.dataDir)

print "Loading data ..."

filelist = [ f for f in listdir(dataDir) if isfile(join(dataDir,f)) ]
count = 1

# read in X and Y training data
for file in natsorted(filelist):
    if file.endswith(".csv") and "with" in file:
        # read .csv data
        data = np.loadtxt(open(dataDir+"/"+file,"rb"),delimiter=",")

        mean = np.mean(data)
        print mean
        if mean > 2 or mean < 0.01:
            filename = file.replace(".csv","")
            print "Removing file "+filename

            # delete files
            if os.path.isfile(dataDir+"/"+filename+".csv"):
                os.remove(dataDir+"/"+filename+".csv")

            if os.path.isfile(dataDir+"/"+filename+".png"):
                os.remove(dataDir+"/"+filename+".png")

            if "without" not in file:
                # delete mine files
                fileid = filename.replace("with","")
                if os.path.isfile(dataDir+"/mine"+str(fileid)+".csv"):
                    os.remove(dataDir+"/mine"+str(fileid)+".csv")

        count += 1

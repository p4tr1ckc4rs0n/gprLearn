#!/usr/bin/python

################################################################################
# save-data.py
#
# A simple script to save the final data in to numpy files
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import sys, os
import argparse
from os.path import expanduser
import Image
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import gc
from loading_methods import load_data
sys.dont_write_bytecode = True

#   Usage:
#
#     process-data.py -i <str> -o <str>
#
#   where
#
#     --dataDir <str> this is the directory where the data to be loaded is
#     -d
#

# parse input commands
parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('-d','--dataDir', dest='dataDir', metavar='dataDir', type=str,
                   help='string for data directory', required=True)

args = parser.parse_args()

# switch to correct directory
dataDir = str(args.dataDir)
rootDir = os.path.dirname(os.path.dirname(dataDir))

X_data, Y_data = load_data(rootDir, dataDir, "radargrams", 0)

os.chdir(dataDir)

# save X and Y data to numpy arrays in the stage4 output
np.save("X_data", X_data)
np.save("Y_data", Y_data)

#

#!/usr/bin/python


################################################################################
# list_gen_lmdb.py
#
# This python script creates the list files and lmdb directory necessary for
# caffe. The list files are txt files with a list of images for caffe as well
# as their class.
################################################################################

import os
from os import listdir
import argparse
import shutil

#
#   Usage:
#
#     list_gen.py -i <str>
#
#   where
#     --input <str> this is the directory in which .png files are stored
#     -i
#

parser = argparse.ArgumentParser(description='generate some txt files')

parser.add_argument('-i','--input', dest='dataDir', metavar='dataDir', type=str,
                   help='string for data directory', required=True)

args = parser.parse_args()

# switch to correct directory
dataDir = str(args.dataDir)
os.chdir(dataDir)

# check to see if labels directory exists, if so delete
if os.path.exists(dataDir+"/labels"):
    shutil.rmtree("labels")
# create labels directory
else:
    os.mkdir("labels")

def list_gen(dataDir):

    # change to labels directory
    os.chdir("labels")

    # create list files
    ftest = open("test.txt","w+")
    ftrain = open("train.txt","w+")
    fval = open("validation.txt","w+")

    for filename in os.listdir(dataDir+"/test"):
        if "without" in filename:
            ftest.write("/"+filename + " 0" + '\n')

        elif "with" in filename:
            ftest.write("/"+filename + " 1" + '\n')

    for filename in os.listdir(dataDir+"/train"):
        if "without" in filename:
            ftrain.write("/"+filename + " 0" + '\n')

        elif "with" in filename:
            ftrain.write("/"+filename + " 1" + '\n')

    for filename in os.listdir(dataDir+"/validation"):
        if "without" in filename:
            fval.write("/"+filename + " 0" + '\n')

        elif "with" in filename:
            fval.write("/"+filename + " 1" + '\n')

    ftest.close()
    ftrain.close()
    fval.close()

def main():
    list_gen(dataDir)
    print "----------------------------------------------------"
    print "List files generated"
    print "----------------------------------------------------"
    print ""

if __name__ == "__main__":
    main()

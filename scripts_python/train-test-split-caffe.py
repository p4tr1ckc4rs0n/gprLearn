#!/usr/bin/python

################################################################################
# train-test-split-caffe.py
#
# This python script splits the training and test images in to two
# separate directories by splitting a list of images and moving their
# directories
################################################################################

from sklearn.cross_validation import train_test_split
import os
import shutil
import argparse

#   Usage:
#
#     train-test-split.py -i <str>
#
#   where
#     --input <str> this is the directory in which the .csv and .png files are stored
#     -i
#
#     --output <str> this is the directory in which the .csv and .png files will be moved to
#     -o
#

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

# set up list for images
imagelist = []

# create list of filenames
for file in os.listdir(dataDir):
    if file.endswith(".png"):
        imagelist.append(file)

#  Split radargram images into lists for entry into train/test directories (75/25% train/test split)
all_train, test_list = \
    train_test_split(imagelist, test_size=0.25)

train_list, val_list = \
    train_test_split(all_train, test_size=0.25)

# create directories for train and test data
os.chdir(outputDir)
os.makedirs("train")
os.makedirs("validation")
os.makedirs("test")

# move files to appropriate directories
def archive_data(directory_name,imagelist):

    for image_file in imagelist:
        shutil.copy(dataDir+"/"+image_file, outputDir+"/"+directory_name)

def main():
    archive_data("train",train_list)
    archive_data("validation",val_list)
    archive_data("test",test_list)

    print "----------------------------------------------------"
    print "Train test split completed"
    print "----------------------------------------------------"
    print ""

if __name__ == "__main__":
    main()

#!/bin/bash

#########################################################################################
#
# Pipeline to prep data for input and execution of caffe learning algorithm
#
#########################################################################################

# Path names.
export GPRMAX_HOME=$(pwd)
export PYTHON_SCRIPTS=$(pwd)/scripts_python
export BASH_SCRIPTS=$(pwd)/scripts_bash
export MODELS=$(pwd)/models
cd ..
export CAFFE=$(pwd)/deep-learning/caffe
export CAFFE_TOOLS=$(pwd)/deep-learning/caffe/build/tools

# Define appropriate data directory and target directories
SIM_DIR=/home/pwhc/skycap/DATA_FLAT
UAV_DIR=/home/pwhc/skycap/DATA_UAV
TARGET_DIR=/tmp/stage5
MEAN_DIR=$TARGET_DIR/mean
ARCH_DIR=$TARGET_DIR/architecture

# remove contents of previous caffe run
rm -rf /tmp/stage5/*
rm -rf /tmp/mean/*
rm -rf /tmp/architecture/*

# remove directories from previous caffe run
rmdir $ARCH_DIR
rmdir $TARGET_DIR
rmdir $MEAN_DIR
rmdir $ARCH_DIR

# create directories 
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating directory: $TARGET_DIR"
    mkdir $TARGET_DIR
    cd $TARGET_DIR
    mkdir saved_models
    cd ..
else
    echo "$TARGET_DIR already exists! Exiting..."
    exit
fi

if [ ! -d "$MEAN_DIR" ]; then
    echo "Creating directory: $MEAN_DIR"
    mkdir $MEAN_DIR
else
    echo "$MEAN_DIR already exists! Exiting..."
    exit
fi

if [ ! -d "$ARCH_DIR" ]; then
    echo "Creating directory: $ARCH_DIR"
    mkdir $ARCH_DIR
else
    echo "$ARCH_DIR already exists! Exiting..."
    exit
fi

################################################################
# Stage 6: Split training and test data
################################################################

echo "###"
echo "Initiate train test split"
echo "###"

$PYTHON_SCRIPTS/train-test-split-caffe.py -i $SIM_DIR -o $TARGET_DIR

################################################################
# Stage 7: Create list files for caffe and lmdb directory
################################################################

echo "###"
echo "Generating list files"
echo "###"

$PYTHON_SCRIPTS/list_gen_lmdb.py -i $TARGET_DIR

################################################################
# Stage 8: Convert imageset to lmdb for caffe
################################################################
## use convert_imageset provided by caffe to turn images to database

echo "###"
echo "Creating caffe database (lmdb)"
echo "###"

$BASH_SCRIPTS/create_lmdb.sh $CAFFE_TOOLS

################################################################
# Stage 9: Create mean image
################################################################
## use convert_imageset provided by caffe to turn images to database

echo "###"
echo "Creating mean image"
echo "###"

$CAFFE_TOOLS/compute_image_mean $TARGET_DIR/train/train-lmdb \
  $MEAN_DIR/mean.binaryproto

################################################################
# Stage 10: Train Test and Validate caffe model
################################################################

python $MODELS/model-caffe.py --train $TRAIN_DIR/train-lmdb --test $TEST_DIR/test-lmdb --val LABELS/validation.txt -c $MODELS/caffe_files -m $MEAN_DIR --uav $UAV_DIR

################################################################
# Stage 11: Save convnet architecture
#################################################################

echo "###"
echo "Saving convnet architecture"
echo "###"

python $CAFFE/python/draw_net.py $MODELS/caffe_files/conv_net_train.prototxt architecure.png
mv architecure.png $ARCH_DIR
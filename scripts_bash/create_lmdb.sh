#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the gprnet train + val data dirs

DATA_DIR=/tmp/stage5
TRAIN_DATA_ROOT=$DATA_DIR/train
VAL_DATA_ROOT=$DATA_DIR/validation
TEST_DATA_ROOT=$DATA_DIR/test

DATABASE=$DATA_DIR/lmdb
DATALIST=$DATA_DIR/labels

# CAFFE directory is provided as an argument
TOOLS=$1

# Set RESIZE=true to resize the images to 64x64. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=64
  RESIZE_WIDTH=64

else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  echo "Set the TEST_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

$TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --gray \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATALIST/train.txt \
    $TRAIN_DATA_ROOT/train-lmdb

echo "Creating test lmdb..."

$TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --gray \
    --shuffle \
    $TEST_DATA_ROOT \
    $DATALIST/test.txt \
    $TEST_DATA_ROOT/test-lmdb

echo "Done."

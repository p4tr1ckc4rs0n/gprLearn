#!/bin/bash

#########################################################################################
# Pipeline to simulate, archive and process gpr data
#########################################################################################

# Path names
export GPRMAX_HOME=$(pwd)
export GPRMAX=$(pwd)/gprMax
export PYTHON_SCRIPTS=$(pwd)/scripts_python


# Define root directory
export ROOT_DIR=$(pwd)/data

# Define location of temporary directories
# which hold the output from each stage of the pipeline
export STAGE1_OUTPUT_DIR=$ROOT_DIR/stage1
export STAGE2_OUTPUT_DIR=$ROOT_DIR/stage2
export STAGE3_OUTPUT_DIR=$ROOT_DIR/stage3

# Check and create root data directory
if [ ! -d "$ROOT_DIR" ]; then
    echo "Creating root directory: $ROOT_DIR"
    mkdir $ROOT_DIR
    # Create directories for main stages of pipeline
    if [ ! -d "$STAGE1_OUTPUT_DIR" ]; then
        echo "Creating directory: $STAGE1_OUTPUT_DIR"
        mkdir $STAGE1_OUTPUT_DIR
    else
        echo "$STAGE1_OUTPUT_DIR already exists! Exiting..."
        exit
    fi
    if [ ! -d "$STAGE2_OUTPUT_DIR" ]; then
        echo "Creating directory: $STAGE2_OUTPUT_DIR"
        mkdir $STAGE2_OUTPUT_DIR
    else
        echo "$STAGE2_OUTPUT_DIR already exists! Exiting..."
        exit
    fi
    if [ ! -d "$STAGE3_OUTPUT_DIR" ]; then
        echo "Creating directory: $STAGE3_OUTPUT_DIR"
        mkdir $STAGE3_OUTPUT_DIR
    else
        echo "$STAGE3_OUTPUT_DIR already exists! Exiting..."
        exit
    fi
else
    echo "$ROOT_DIR already exists!"
    echo "Type name of new root directory to be created followed by [ENTER]: "
    read USER_INPUT
    NEW_DIR=$(pwd)/$USER_INPUT
    # Re-assign directories that hold output from each stage of the pipeline
    export STAGE1_OUTPUT_DIR=$NEW_DIR/stage1
    export STAGE2_OUTPUT_DIR=$NEW_DIR/stage2
    export STAGE3_OUTPUT_DIR=$NEW_DIR/stage3
    if [ ! -d $USER_INPUT ]; then
        echo "Creating alternate root directory: $USER_INPUT"
        mkdir $NEW_DIR
        # Create directories for main stages of pipeline
        if [ ! -d "$STAGE1_OUTPUT_DIR" ]; then
            echo "Creating directory: $STAGE1_OUTPUT_DIR"
            mkdir $STAGE1_OUTPUT_DIR
        else
            echo "$STAGE1_OUTPUT_DIR already exists! Exiting..."
            exit
        fi
        if [ ! -d "$STAGE2_OUTPUT_DIR" ]; then
            echo "Creating directory: $STAGE2_OUTPUT_DIR"
            mkdir $STAGE2_OUTPUT_DIR
        else
            echo "$STAGE2_OUTPUT_DIR already exists! Exiting..."
            exit
        fi
        if [ ! -d "$STAGE3_OUTPUT_DIR" ]; then
            echo "Creating directory: $STAGE3_OUTPUT_DIR"
            mkdir $STAGE3_OUTPUT_DIR
        else
            echo "$STAGE3_OUTPUT_DIR already exists! Exiting..."
            exit
        fi
    fi
fi

cp /home/pwhc/skycap/gprlearn/data_33/stage1/with0_.in /home/pwhc/skycap/gprlearn/re-run/stage1

echo "###"
echo "Creating simulated environments"
echo "###"

export number_iters=95

cd $GPRMAX

# Call GPRMAX to process each input file
for i in $( ls $STAGE1_OUTPUT_DIR/*.in ); do
    echo "Running model:" $i
    python3 -m gprMax $i -n $number_iters
done

#########################################################################################
# Stage 4: Compile individual A-scans into B-scan file
#########################################################################################

echo "###"
echo Compiling gpr files
echo "###"

# Call GPRMAX to compile each output file
for i in $( ls $STAGE1_OUTPUT_DIR/*.in ); do
    file=$i
    filename="${file%.*}"
    echo "Compiling model:" $i
    python3 -m tools.outputfiles_merge $filename $number_iters
done

mv $STAGE1_OUTPUT_DIR/*.out $STAGE2_OUTPUT_DIR

#########################################################################################
# Stage 5: Convert B-scan into numpy array, normalise, plot and save
#########################################################################################

echo "###"
echo "Saving B-scan radargrams"
echo "###"

for i in $( ls $STAGE2_OUTPUT_DIR/*.out ); do
    python -m tools.plot_Bscan $i Ez $STAGE3_OUTPUT_DIR
done

echo "Data generation complete"
echo "###"

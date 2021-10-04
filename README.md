## GprLearn Overview

24 May 2016

Patrick Carson
Max Jacobs
Ken Williams

This file: README.txt
REPO: https://bitbucket.org/dgorissen/gprlearn

### 1. INTRODUCTION

The code in this repository is to create a framework for automatically generating
ground penetrating radar data for the training of a convolutional neural net in 
order to accurately classify the presence of a buried landmine.

The data consists of 2D simulated ground penetrating radar B-scan images
over various complex soil environments. Soil dielectric and geometric properties
are accurately modelled to mirror that of real world soil environment. Landmines
are then placed at random within the soil model.

The ultimate purpose of this project is to prepare a classification system through 
creating realistic training data so as to train a convolution neural network to be 
able to automatically recognise and locate landmine signatures in simulated B-scan 
gpr images. Real UAV collected gpr data will then be used to test it's
real world applicability.

The files in this repository are described below. The framework was developed and
runs on Ubuntu Linux.

2. INSTALLATION:

- gprMax 
This is an open source piece of software that simulates electromagnetic wave propagation
in various user defined mediums. For installation see http://gprmax.readthedocs.io/en/latest/
and get help from https://groups.google.com/forum/?hl=en-GB#!forum/gprmax.

Input files (*.in) specify a terrain, the simulator then returns multiple output files (*.out)
which are then merged to a single output file. The number of individual output files is equal 
to the number of specified traces. This merged output file is then post-proccessed and 
plotted to create the resultant gpr B-scans image.

Command line tags exist to allow the user to produce a geometry file (*.vti) of the input soil
model. The geometry view shows the geometric makeup of the soil model, location of any sources 
and recievers (gpr antenna) and in our case the location of the landmine. Installing the
open-source piece of software Paraview is necessary to visualise the geometry file.
See http://www.paraview.org/

- Caffe 
Caffe is a popular open-source platform for creating, training, evaluating and deploying deep 
neural networks. Caffe is seen as one of the leading libraries for image classification and 
other functions of computer vision. Caffe is used in this project to create the convolutional
neural network used to classify and locate landmine signatures in the simulated gpr data.

Install caffe according to these instructions http://hanzratech.in/2015/07/27/installing-caffe-on-ubuntu.html

### 2. GETTING STARTED

- git clone https://<your-username>@bitbucket.org/dgorissen/gprlearn.git

From here, if all prerequisites are installed, the pipeline should run.

Prerequisites:
  - Ubuntu Linux
  - python2.7 
  - python3 for gprMax
  - caffe

### 3. OVERVIEW OF PROJECT

**Some files and directories have been left out since they were used for the old version of gprMax**

gprlearn/
  gprMax/
  models/
    caffe_files/
    model-caffe.py
  scripts_bash/
    create_lmdb.sh
  scripts_python/
    gpr-gen-new.py
    post_process.py
    list_gen_lmdb.py
    train-test-split-caffe.py
    localise.py
  README.TXT
  pipeline-gen-new.sh
  individual-pipeline.sh
  pipeline-caffe.sh

### 4. DATA GENERATION PIPELINE

Within the Linux shell run the pipeline with the command:

    $  ./pipeline-gen-new.sh

This is the bash shell script which runs the data generation part of the project. In this 
script, 3 directories are created to store successive files associated with the data generation 
process - as can be seen in the bash shell script itself. Essentially,

    STAGE 1. Create directory structure to hold data from each stage of the pipeline e.g,

      */STAGE1/
      */STAGE2/
      */STAGE3/

    STAGE 2.

	    RUN  gen-gpr-new.py 

          //  This python script generates random data
              in the form of *.in files. The number of files
              generated depends on the command-line arguments
              passed to the script. See the script comments
              for more details. The files are written to the
              $STAGE1_OUTPUT_DIR

    STAGE 3.

	    RUN  python3 -m gprMax

          //  Run gprMax on each *.in file generated.
              For each *.in file gprMax generates multiple *.out
              files equal to the number of specified gpr traces.
              The files are written to the $STAGE1_OUTPUT_DIR.

    STAGE 4.

	    RUN  python3 -m tools.outputfiles_merge

          //  Run this gprMax script on all *.out files for each
              *.in file generated. All the individual *.out files
              for the corresponding *.in file are merged to
              creat one *.out file for each *.in file. These
              are written to the $STAGE2_OUTPUT_DIR

    STAGE 5.

      RUN python3 post_process.py

          //  This script is run on each merged *.out file,
              a post-processing technique is employed to remove clutter,
              the *.out file is then plotted and saved as a greyscale
              radargram in $STAGE3_OUTOUT_DIR


The pipeline-gen-new.sh script ends at this point.

### 5. CLASSIFICATION PIPELINE

Within the Linux shell run the pipeline with the command:

    $  ./pipeline-caffe.sh

This is the bash shell script that takes the generated gpr data, pre-processes it and then passes
it through the convolutional neural network for training, testing and validation. In this script
1 overall directory is created with inner directories being created for each part of the classification
process. Essentially,

    STAGE 6. Create directory structure to store necessary files for running caffe model

      */STAGE5/
        labels/
        mean/
        architecture/
        saved_models/
        test/
        train/
        validation/

    STAGE 7.

      RUN  train-test-split-caffe.py

          //  This script takes the previously generated data
              and splits it up into training, testing and 
              validation data for the caffe model. Data is
              then moved to the appropriate directories.

    STAGE 8.

      RUN  list_gen_lmdb.py

          //  This script creates the list files and lmdb
              directory necessay for caffe. Text files are
              created that contain the image path and their
              associated class. 

    STAGE 10. 

      RUN create_lmdb.sh

          //  This bash script creates the convnet memory-mapped
              lmdb inputs necessary for caffe.

    STAGE 11.

      RUN compute_image_mean.sh

          //  This file generates an overall mean image from all
              the training data, necessary for pre-processing
              the radargrams before they are passed to the caffe
              model.

    STAGE 12. 

      RUN model-caffe.py

          //  This script contains the caffe convolutional neural
              network. All of the previously prepared data is passed
              into the model for training, testing and validation. Train
              loss curves are plotted along with a ROC curve and confusion 
              matrix - these can be saved if the user choses so. For Localising 
              the landmine signature refer to localisation.py.

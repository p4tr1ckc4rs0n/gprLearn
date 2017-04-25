#!/usr/bin/python

################################################################################
# gpr-gen-noise.py
#
# This file generates the .in files used by the OLD gprMax to create
# radargrams. It takes a number of inputs for specifying soil properties, these
# should be altered with care as the old gprMax software is sensitive and can
# produce erroneous results
################################################################################

from __future__ import division
import argparse
import datetime
import random
import os
import numpy as np

#   Usage:
#
#     gpr-codegen.py --fileid <str> --name <str> [ --frequency <int> ] [ --with <int> ] [ --envir <str> ] [ --without <int> ] [ --mine_type <str> ] [ --out <str> ]
#
#     where:
#
#     --name <str> will be the root prefix of the Model name and
#     -n           used for the filenames to be created.
#                  The --name parameter is required.
#
#     --fileid <str> id of file
#
#     --frequency <int> will be the choice of GPR frequency used
#     -f    	   in the Model (500Mhz or 1000MHz). The --frequency is
#                  optional. If it is not used then the defulat value of
#                  '500MHz' is chosen.
#
#     --with <int> will be the number of randomly generated input files
#     -w           which contain exactly 1 landmine.
#                  The --with parameter is optional. If it does not
#                  appear then the default value of '1' is used.
#
#     --without <int>  will be the number of randomly generated input files
#     -x           which do not contain a landmine.
#                  The --without parameter is optional. If it does not
#                  appear then the default value of '1' is used.
#
#     --mine_type <str> will be the choice of landmine. Enter either
#     -mt         "anti-personnel", "anti-tank" or "minimum_metal" for
#                 desired mine type.
#
#     --envir <str> will be the choice of subsurface. Enter either "easy"
#     -e          for dry sandy environment, "medium" for damp sany environment
#                 or "hard" for damp clay environment. --envir is required
#
#     --out <str> will specify the output directory where the files are to
#     -o          be written.
#                 The --o parameter is optional. If it does not
#                 appear then the default behaviour is to write to
#                 the current directory.
#
#   Examples
#
#   (1)     gpr-codegen.py --name test --frequency 1000 --with 1 --mine_type anti-personnel
#
#   This will generate 1 input file for gprmax with 1 anti-personnel landmine a GPR
#   frequency of 1000MHz and is called 'test0.in' which will be written
#   to the current directory.
#
#   (2)     gpr-codegen.py --name train --without 1
#
#   This will generate 1 input file for gprmax without any landmine, a
#   default GPR frequency of 500MHz and is called 'train0.in' which will
#   be written to the current directory.
#
#   (3)     gpr-codegen.py --name trainingdata --frequency 1000 --with 10 --mine_type anti-tank --out training
#
#   This will generate 10 input files for gprmax each with 1 anti-tank landmine, a
#   GPR frequency of 1000MHz and are called 'trainingdata0.in','
#   trainingdata1.in', 'trainingdata2.in' ...'trainingdata9.in' which will
#   be written to the 'training' directory.
#
#   (4)     gpr-codegen.py --name testdata --without 40 --out test_dir
#
#   This will generate 40 input files for gprmax each without any
#   landmine a default GPR frequency of 500MHz and are called   'testdata0.in',
#   'testdata1.in', 'testdata2.in' ... 'testdata39.in' which will be written to
#   the 'test_dir' directory.
#
#   SHORT-FORMS
#       The command line options also have short-forms,
#           --name, -n
#           --frequency, -f
#           --envir, -e
#           --with, -w
#           --without, -x
#           --mine_tpye, -mt
#           --out, -o
#
#   Here are examples 1,2,3 and 4 written using their short-forms,
#
#   (1)     gpr-codegen.py -n test -w 1 -f 1000 -mt anti-personnel
#   (2)     gpr-codegen.py -n train -x 1
#   (3)     gpr-codegen.py -n trainingdata -f 1000 -w 10 -mt anti-tank -o training
#   (4)     gpr-codegen.py -n testdata -x 40 -o test_dir
#
#   NOTE: Currently you cannot use both the --with and --without switches
#   in the same command-line. This is because the program only generates
#   .in files containing landmines or .in files containing no landmines,
#   it does not generate a mixture of both. This may change in a future
#   release.
#
# Global variables (and default values)
#
filesToBeCreated = 0
outputDir = "."      									# default output to current directory.

#
# Constants
#
LINE = "---------------------------------------------------------------------"
NL = "\n"

#
# GPRMAX values
#
analysisVal = 115       									# number of new model runs
domainDepthVal = 0.7 									# depth of model in y direction (m)
domainDistanceVal = 2.5 								# distance in x direction of model (m)

#
# Parse command-line arguments
#
parser = argparse.ArgumentParser(description="Generate *.in input files for gprmax")
parser.add_argument('-n', '--name',metavar='name', required=True,
                    dest='name', action='store',
                    help='Model name')

parser.add_argument('-f','--frequency',help='Choice of GPR frequency (500MHz or 1000MHz)',
                    default=500,type=int)

parser.add_argument('-e', '--envir', dest='envir', action='store',type=str,
                    help='choice of environment',required=True)

parser.add_argument('-w', '--with', dest='filesWith', action='store',
                    help='Number of WITH files to be created')

parser.add_argument('-x', '--without', dest='filesWithout', action='store',
                    help='Number of WITHOUT files to be created')

parser.add_argument('-mt','--mine_type', metavar='mine_type', dest="mine_type",
                    action="store", help='enter mine type')

parser.add_argument('-o', '--out', dest='outputDir', action='store',
                    help='output directory')

parser.add_argument('-fi', '--fileid', dest='fileid', action='store',
                    help='id of file')

args = parser.parse_args()

#
# Cannot use --with and --without switches at the same time
# (see 'NOTE' in main comments above)
#
if (args.filesWith) and (args.filesWithout):
    print "ERROR -> Cannot use --with and --without switches at same time."
    exit(0)

#
# Cannot use --without and --mine_type switches at the same
# time for obvious reasons
#
if (args.filesWithout) and (args.mine_type):
    print "Error -> Cannot use --without and --mine_type at the same time."
    exit(0)

if (args.filesWith) and not (args.mine_type):
    print "Error -> Please include --mine_type switch"
    exit(0)

if (args.filesWith):
    filesToBeCreated = int(args.filesWith)

else:
    filesToBeCreated = int(args.filesWithout)

if (args.outputDir):
    outputDir = args.outputDir

if (args.fileid):
    fileid = args.fileid

envir = args.envir

#################################
# DEFINE HETERGENEOUS SUBSURFACE
#################################

media_dir = str(os.path.realpath("text_files/media.txt"))

def subsurface_gen(envir):

    easy_range = []
    medium_range = []
    hard_range = []

    m = open(media_dir,"r")
    contents = m.readlines()
    for line in contents:
        if "easy" in line:
            easy_range.append(line.split()[-1])
        elif "medium" in line:
            medium_range.append(line.split()[-1])
        elif "hard" in line:
            hard_range.append(line.split()[-1])

    if envir == "easy":

        f.write("#box: 0.0 0.0 " + str(domainDistanceVal) + " 0.6"
            + " dry_sand" + NL)

        for num in xrange(50):

            ll_x = random.uniform(0,2.2)
            ll_y = random.uniform(0,0.4)
            ur_x = ll_x + random.uniform(0.1,0.2)
            ur_y = ll_y + random.uniform(0.1,0.2)

            dec = 2

            # check coords are okay
            if np.around(ur_x, decimals=dec) > np.around(ll_x, decimals=dec) and np.around(ur_y, decimals=dec) > np.around(ll_y, decimals=dec):
                f.write("#box: " + str(ll_x) + " " + str(ll_y) +
                  " " + str(ur_x) + " " + str(ur_y) + " " + str(random.choice(easy_range)) + NL)

    elif envir == "medium":

        f.write("#box: 0.0 0.0 " + str(domainDistanceVal) + " 0.6"
            + " damp_sand" + NL)

        for num in xrange(250):

            ll_x = random.uniform(0,2.2)
            ll_y = random.uniform(0,0.4)
            ur_x = ll_x + random.uniform(0.1,0.2)
            ur_y = ll_y + random.uniform(0.1,0.2)

            dec = 2

            # check coords are okay
            if np.around(ur_x, decimals=dec) > np.around(ll_x, decimals=dec) and np.around(ur_y, decimals=dec) > np.around(ll_y, decimals=dec):
                f.write("#box: " + str(ll_x) + " " + str(ll_y) +
                  " " + str(ur_x) + " " + str(ur_y) + " " + str(random.choice(medium_range)) + NL)

    elif envir == "hard":

        f.write("#box: 0.0 0.0 " + str(domainDistanceVal) + " 0.6"
            + " dry_claysoil" + NL)

        for num in xrange(350):

            ll_x = random.uniform(0.0,2.2)
            ll_y = random.uniform(0.0,0.4)
            ur_x = ll_x + random.uniform(0.1,0.2)
            ur_y = ll_y + random.uniform(0.1,0.2)

            dec = 2

            # check coords are okay
            if np.around(ur_x, decimals=dec) > np.around(ll_x, decimals=dec) and np.around(ur_y, decimals=dec) > np.around(ll_y, decimals=dec):
                f.write("#box: " + str(ll_x) + " " + str(ll_y) +
                  " " + str(ur_x) + " " + str(ur_y) + " " + str(random.choice(hard_range)) + NL)

    else:
        print 'Error -> Please enter either:\n1. "easy"\n2. "medium"\n3. "hard"'
        exit(0)

def surface_gen():
       ll_x = 0.0
       ll_y = 0.3
       ur_y = domainDepthVal
       width = 0.2
       ur_x = ll_x + width

       f.write("#box: " + str(ll_x) + " " + str(ll_y) + " "
    		    + str(ur_x) + " " + str(ur_y) +
    		    " free_space" + NL)

       for num in xrange(10):
           delta_h = random.gauss(0,0.1)/5
           ll_x += width
           ll_y += delta_h

           f.write("#box: " + str(ll_x) + " " + str(ll_y) + " "
               + str(ll_x + width) + " " + str(ur_y) +

               " free_space" + NL)

#################################
# DEFINE MINE TYPE
#################################

def mine_generation(mine_type, filename, tx_steps):

    if mine_type == "anti_tank":
        # Soviet TM-46 anti-tank mine (approx)
        landmineBodyWidth1 = 0.305
        landmineBodyHeight1 = 0.108

        rand_x = random.uniform(0,0.8) * domainDistanceVal
        rand_y = random.uniform(0,0.5) * domainDepthVal
        base_ll_x = rand_x
        base_ll_y = rand_y
        base_ur_x = base_ll_x + landmineBodyWidth1
        base_ur_y = base_ll_y + landmineBodyHeight1
        base_medium_id = " pec "

        f.write("-- anti-tank landmine" + NL)
        f.write("#box: " + str(base_ll_x) + " " + str(base_ll_y) + " "
          + str(base_ur_x) + " " + str(base_ur_y) + " " + base_medium_id + NL)

        upper_ll_x = base_ll_x + (landmineBodyWidth1 / 3.0)
        upper_ll_y = base_ll_y + (landmineBodyHeight1 / 3.0)
        upper_ur_x = base_ur_x - (landmineBodyWidth1 / 3.0)
        upper_ur_y = base_ur_y + (landmineBodyHeight1 / 10.0)
        upper_medium_id = " pec "

        f.write("#box: " + str(upper_ll_x) + " " + str(upper_ll_y) + " "
          + str(upper_ur_x) + " " + str(upper_ur_y) + " " + upper_medium_id + NL)


    elif mine_type == "anti_personnel":
        # Italian AUPS bakelite anti-personnel mine
        landmineBodyWidth2 = 0.102
        landmineBodyHeight2 = 0.036

        rand_x = random.uniform(0,0.96) * domainDistanceVal
        rand_y = random.uniform(0,0.5) * domainDepthVal
        base_ll_x = rand_x
        base_ll_y = rand_y
        base_ur_x = base_ll_x + landmineBodyWidth2
        base_ur_y = base_ll_y + landmineBodyHeight2
        base_medium_id = " pec "

        f.write("-- anti-personnel landmine" + NL)
        f.write("#box: " + str(base_ll_x) + " " + str(base_ll_y) + " "
          + str(base_ur_x) + " " + str(base_ur_y) + " " + base_medium_id + NL)


    elif mine_type == "minimum_metal":
        # Italian VS-50 minimum metal mine (plastic)
        landmineBodyWidth3 = 0.090
        landmineBodyHeight3 = 0.045

        rand_x = random.uniform(0,0.96) * domainDistanceVal
        rand_y = random.uniform(0,0.5) * domainDepthVal
        base_ll_x = rand_x
        base_ll_y = rand_y
        base_ur_x = base_ll_x + landmineBodyWidth3
        base_ur_y = base_ll_y + landmineBodyHeight3
        base_medium_id = " plastic"

        f.write("-- minimum metal landmine" + NL)
        f.write("#box: " + str(base_ll_x) + " " + str(base_ll_y) + " "
          + str(base_ur_x) + " " + str(base_ur_y) + " " + base_medium_id + NL)

    else:
        print 'Error -> Please enter either:\n1. "anti_tank"\n2. "anti_personnel"\n3. "minimum_metal"'

    # now write to file the location of the landmine
    if (args.filesWith):
        g = open(outputDir + "/" + filename + "_minepos" + ".csv", 'w+');
        g.write(str(base_ll_x)+","+str(base_ll_y)+","+str(base_ur_x)+","+str(base_ur_y)+","+str(tx_steps));
        g.close();

#################################
# START FOR LOOP
#################################

for fileCount in range(0, filesToBeCreated):

    #
    # Set filename
    #
    fname = args.name

    if (args.fileid):
        fname = fname + fileid
    else:
        fname = fname + str(fileCount)

    #
    # Open the output file and write the header section
    #
    f = open(outputDir + "/" + fname + ".in", 'w+')
    f.write(NL)
    f.write("Filename: " + fname + ".in" + NL)
    f.write("Generated by gpr-gen-noise.py: " + str(datetime.datetime.now()) + NL)
    f.write(LINE + NL)

    #
    # Provide location and name of media file
    # Media file contains available mediums
    # Provide number of available mediums
    #
    f.write("#media_file: " + str(media_dir) + NL)
    f.write("#number_of_media: 80" + NL)
    f.write(LINE + NL)

    #
    # Specify the size of the domain.
    #
    f.write("#domain: " + str(domainDistanceVal) + " " + str(domainDepthVal) + NL) 		# dimensions of model (m)
    f.write("#dx_dy: 0.015 0.015" + NL)							# spatial step in x and y direction (m) - smallest wavelength to be resolved/10
    f.write("#time_window: 12e-9" + NL)	 							# total required simulatied time (s) - related to the length of each trace and therefore depth to be reached
    f.write(LINE + NL)

    #
    # Generate hetergeneous subsurface
    # Call gen_subsurface function
    #
    subsurface_gen(envir)

    #
    # Simulate rough surface
    #
    # surface_gen()


    #
    # Do we add a landmine in this image?
    # If so then we add it to a random x, y location
    # within the domain.
    # Call mine_generation function
    #
    if (args.mine_type):
        mine_type = args.mine_type
        mine_generation(mine_type, fname, 0.02)
        f.write(LINE + NL)
    else:
        f.write(LINE + NL)

    #
    # Specify the GPR line source
    #
    f.write("#line_source: 1.0 " + str(args.frequency) + "e6"+ " ricker MyLineSource" + NL);    # GPR frequency (MHz)
    f.write(LINE + NL);

    #
    # Define the Analysis section.
    #
    f.write("#analysis: " + str(analysisVal) + " " + fname + ".out b" + NL);			# number of new model runs (analysisval)
    f.write("#tx: 0.0875 0.65 MyLineSource 0.0 12e-9" + NL); 					# location of GPR transmitter
    f.write("#rx: 0.1125 0.65" + NL); 								# location of GPR reciever
    f.write("#tx_steps: 0.02 0.0" + NL); 							# advance coordinates of tx transmitter
    f.write("#rx_steps: 0.02 0.0" + NL); 							# advance coordinates of rx reciever
    f.write("#end_analysis: " + NL);
    f.write(LINE + NL);

    #
    # Define a geometry file and model title.
    #
    f.write("#geometry_file: " + fname + ".geo"+ NL);
    f.write("#title: " + fname.upper() + " Model 1"+ NL);
    f.write("#messages: y"+ NL);

    #
    # Close the file when finished.
    #
    f.close();

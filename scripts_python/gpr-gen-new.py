#!/usr/bin/python

################################################################################
# gpr-gen-new.py
#
# This file generates the .in files used by the NEW gprMax to create
# radargrams. It takes a number of inputsl: with/without mine, RADAR frequency
# and mine type.
#
################################################################################

from __future__ import division
import argparse
import datetime
import random
import os
import numpy as np
import math
import time

#   Usage:
#
#     gpr-gen-new.py --name <str> [ --frequency <int> ] [ --with <int> ] [ --without <int> ] [ --mine_type <str> ] [ --out <str> ]
#
#     where:
#
#     --name <str> will be the root prefix of the Model name and
#     -n           used for the filenames to be created.
#                  The --name parameter is required.
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
#     --out <str> will specify the output directory where the files are to
#     -o          be written.
#                 The --o parameter is optional. If it does not
#                 appear then the default behaviour is to write to
#                 the current directory.
#
#   Examples
#
#   (1)     gpr-gen-new.py --name test --frequency 1 --with 1 --mine_type anti-personnel
#
#   This will generate 1 input file for gprmax with 1 anti-personnel landmine a GPR
#   frequency of 1000MHz and is called 'test0.in' which will be written
#   to the current directory.
#
#   (2)     gpr-gen-new.py --name train --without 1
#
#   This will generate 1 input file for gprmax without any landmine, a
#   default GPR frequency of 500MHz and is called 'train0.in' which will
#   be written to the current directory.
#
#   (3)     gpr-gen-new.py --name trainingdata --frequency 1 --with 10 --mine_type anti-tank --out training
#
#   This will generate 10 input files for gprmax each with 1 anti-tank landmine, a
#   GPR frequency of 1000MHz and are called 'trainingdata0.in','
#   trainingdata1.in', 'trainingdata2.in' ...'trainingdata9.in' which will
#   be written to the 'training' directory.
#
#   (4)     gpr-gen-new.py --name testdata --without 40 --out test_dir
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
#           --with, -w
#           --without, -x
#           --mine_tpye, -mt
#           --out, -o
#
#   Here are examples 1,2,3 and 4 written using their short-forms,
#
#   (1)     gpr-gen-ew.py -n test -w 1 -f 1 -mt anti-personnel
#   (2)     gpr-gen-new.py -n train -x 1
#   (3)     gpr-gen-new.py -n trainingdata -f 1 -w 10 -mt anti-tank -o training
#   (4)     gpr-gen-new.py -n testdata -x 40 -o test_dir
#
#   NOTE: Currently you cannot use both the --with and --without switches
#   in the same command-line. This is because the program only generates
#   .in files containing landmines or .in files containing no landmines,
#   it does not generate a mixture of both. 

#
# Global variables (and default values)
#
filesToBeCreated = 0
outputDir = "."

#
# Constants
#
NL = "\n"

#
# Parse command-line arguments
#
parser = argparse.ArgumentParser(description="Generate *.in input files for gprmax")
parser.add_argument('-n', '--name',metavar='name', required=True,
                    dest='name', action='store',
                    help='Model name')

parser.add_argument('-f','--frequency',help='Choice of GPR frequency (500MHz or 1000MHz)',
                    default=0.5,type=float)

parser.add_argument('-w', '--with', dest='filesWith', action='store',
                    help='Number of WITH files to be created')

parser.add_argument('-x', '--without', dest='filesWithout', action='store',
                    help='Number of WITHOUT files to be created')

parser.add_argument('-mt','--mine_type', metavar='mine_type', dest="mine_type",
                    action="store", help='enter mine type')

parser.add_argument('-o', '--out', dest='outputDir', action='store',
                    help='output directory')

parser.add_argument('-r','--rough',dest='rough_surface',action='store',
                    help='define B-scan with rough surface',required=True)

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
    fileid = args.fileid+"_"

rough_tag = str((args.rough_surface))

#
# GPRMAX domain values
#
domainDepthVal = 1.0
domainDistanceVal = 2.0                               

#################################
# DEFINE MINE TYPE
#################################

def mine_generation(mine_type):
            
    if mine_type == "anti_tank":
        
        # Soviet TM-46 anti-tank mine (approx)
        landmineBodyWidth1 = 0.305
        landmineBodyHeight1 = 0.108

        # define landmine coordinates
        base_ll_x = round(random.uniform(0.1,1.6),3)
        base_ll_y = round(random.uniform(0.3,0.5),3)
        base_ur_x = base_ll_x + landmineBodyWidth1
        base_ur_y = base_ll_y + landmineBodyHeight1
        
        # write landmine coordinates to input file
        f.write("-- anti-tank landmine" + NL)
        f.write("#box: " + str(base_ll_x) + " " + str(base_ll_y) + " 0 "
          + str(base_ur_x) + " " + str(base_ur_y) + " 0.001 pec" + NL)

        # further landmine coordinates
        upper_ll_x = round(base_ll_x + (landmineBodyWidth1 / 3.0),3)
        upper_ll_y = base_ur_y
        upper_ur_x = round(base_ur_x - (landmineBodyWidth1 / 3.0),3)
        upper_ur_y = round(base_ur_y + (landmineBodyHeight1 / 10.0),3)
        
        # write landmine coordinates to input file
        f.write("#box: " + str(upper_ll_x) + " " + str(upper_ll_y) + " 0 "
          + str(upper_ur_x) + " " + str(upper_ur_y) + " 0.001 pec" + NL)
        
        #centre_x = base_ur_x - (landmineBodyWidth1/2)
        #centre_y = upper_ur_y - (landmineBodyHeight1/2)
                
    else:
        raise ValueError('type "anti_tank"')

    #return [centre_x,centre_y]

#################################
# ADD NOISE 
#################################

def add_noise(no_objects):
    
    width = 0.305
    height = 0.108

    for num in range(int(no_objects/3)):
        
        diameter = round(random.uniform(0.01,0.025),3)    
        crand_x = round(random.uniform(0.05,1.9),2)
        crand_y = round(random.uniform(0.01,0.2),2)

        f.write("#sphere: " + str(crand_x) + " " + str(crand_y) + " 0.001 " + str(diameter)\
                + " shale" + NL)
        
        '''
        if argv:
            if abs(crand_x-argv[0][0]) < width and abs(crand_y-argv[0][1]) < height:
                continue
            f.write("#sphere: " + str(crand_x) + " " + str(crand_y) + " 0.001 " + str(diameter)\
                + " shale" + NL)
        else:
            f.write("#sphere: " + str(crand_x) + " " + str(crand_y)+ " 0.001 " + str(diameter)\
                + " shale" + NL)
        '''


    for num in range(int(no_objects/3)):
        
        diameter = round(random.uniform(0.01,0.02),3)    
        crand_x = round(random.uniform(0.01,1.95),2)
        crand_y = round(random.uniform(0.2,0.4),2)
        
        f.write("#sphere: " + str(crand_x) + " " + str(crand_y) + " 0.001 " + str(diameter)\
                + " shale" + NL)

        '''
        if argv:
            if abs(crand_x-argv[0][0]) < width and abs(crand_y-argv[0][1]) < height:
                continue
            f.write("#sphere: " + str(crand_x) + " " + str(crand_y) + " 0.001 " + str(diameter)\
                + " shale" + NL)
        else:
            f.write("#sphere: " + str(crand_x) + " " + str(crand_y)+ " 0.001 " + str(diameter)\
                + " shale" + NL)
        '''

    for num in range(int(no_objects/3)):
        
        diameter = round(random.uniform(0.01,0.015),3)    
        crand_x = round(random.uniform(0.01,1.95),2)
        crand_y = round(random.uniform(0.4,0.6),2)

        f.write("#sphere: " + str(crand_x) + " " + str(crand_y) + " 0.001 " + str(diameter)\
                + " shale" + NL)
        
        '''
        if argv:
            if abs(crand_x-argv[0][0]) < width and abs(crand_y-argv[0][1]) < height:
                continue
            f.write("#sphere: " + str(crand_x) + " " + str(crand_y) + " 0.001 " + str(diameter)\
                + " shale" + NL)
        else:
            f.write("#sphere: " + str(crand_x) + " " + str(crand_y)+ " 0.001 " + str(diameter)\
                + " shale" + NL)
        ''' 

#################################
# START FOR LOOP
#################################

for fileCount in xrange(0, filesToBeCreated):

    fname = args.name

    if (args.fileid):
        fname = fname + fileid
    else:
        fname = fname + str(fileCount)

    # calculate soil bulk density
    bulk_density = str(round(random.uniform(1.1,1.66),3))

    # calculate sand particle density
    particle_density = str(round(random.uniform(2,2.8),3))

    #
    # open the output file and write gprMax commands
    #
    f = open(outputDir + "/" + fname + ".in", 'w+')
    f.write("--Generated by gpr-gen-new.py: " + str(datetime.datetime.now()) + NL)
    f.write("#title: " + fname + ".in" + NL)
    f.write("#domain: " + str(domainDistanceVal) + " " + str(domainDepthVal) + " 0.001" + NL)
    f.write("#dx_dy_dz: 0.002 0.002 0.001" + NL)                          
    f.write("#time_window: 16e-9" + NL)                                                        
    f.write("#pml_cells: 10 10 0 10 10 0" + NL)
    f.write("#waveform: ricker 1 " + str(args.frequency) + "e9"+ " my_ricker" + NL); 
    f.write("#hertzian_dipole: z 0.05 0.8 0 my_ricker"+ NL)                  
    f.write("#rx: 0.09 0.8 0" + NL)        
    f.write("#src_steps: 0.02 0 0" + NL)                       
    f.write("#rx_steps: 0.02 0 0" + NL)
    f.write("#material: 15.0 0.0 1.0 0.0 shale" + NL)
    f.write("#num_threads: 2" + NL)
    
    #    
    # define soil environment; x layers of soil
    #
    f.write("#soil_peplinski: 0.7 0.3 " + bulk_density + " " + particle_density + " 0.1 0.25 subsurface" + NL)
    f.write("#fractal_box: 0 0 0 2 0.7 0.001 3 1 1 0 50 subsurface soil_box 5" + NL)
    
    #
    # add rough surface
    #
    if rough_tag == 'y':
        f.write("#add_surface_roughness: 0 0.7 0 2.0 0.7 0.001 2.0 1 1 0.65 0.71 soil_box" + NL)
    elif rough_tag == 'n':
        continue
    #
    # Do we add a landmine in this image?
    # If so then we add it to a random x, y location
    # within the domain.
    # Call mine_generation function
    #
    if (args.mine_type):
        mine_type = args.mine_type
        print mine_type
        # add noise
        add_noise(150)
        mine_generation(mine_type)
    else:
        add_noise(150)

    # geometry file
    #f.write("#geometry_view: 0 0 0 2 1 0.001 0.002 0.002 0.001 mine_test n")

    #
    # Close the file when finished.
    #
    f.close();

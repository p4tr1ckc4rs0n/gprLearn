#!/usr/bin/env python

import argparse
import os
import numpy as np
import h5py
import matplotlib.pylab as plt
from pylab import rcParams

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots A-scan figure', usage='')

parser.add_argument('-i','--inputfile',dest='file_name',required=True,action='store',help='name of input file including path')
parser.add_argument('-f','--field',dest='field',default='Ez',help='name of field to be plotted, i.e. Ex, Ey, Ez')
parser.add_argument('-d','--directory',dest='out_dir',required=True,action='store',help='directory in which to save figure')

args = parser.parse_args()

# assign input arguments to variables
file_name = args.file_name
field = args.field
out_dir = args.out_dir

# open h5py file and convert to numpy array
path = '/rxs/rx1'
f = h5py.File(file_name, 'r')
data = f[path + '/' + field]
arr = np.array(data)

if arr.ndim != 1:
	print "exiting, data needs to be 1D"
	exit(0)

# convert x-axis to time
time = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt'])
time = time / 1E-9

# set plotting params
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13 

# plot A-scan
plt.plot(time,arr,'red',linewidth=2)
plt.axvspan(0.5,1.5,color='blue',alpha=0.5)
plt.axvspan(1.8,2.6,color='green',alpha=0.5)
plt.title('A-scan',fontsize=20)
plt.xlabel('Time [ns]',fontsize=20)
plt.ylabel('Field Strength [V/m]',fontsize=20)
plt.grid()
plt.savefig(out_dir+'/'+'A-scan',format='png')

print "A-scan plotted and saved in current directory"

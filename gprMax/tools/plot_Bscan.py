# Copyright (C) 2015: The University of Edinburgh
#            Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
#from gprMax.exceptions import CmdInputError

################################################################################
# UPDATES
#
# This file normalises the radargram to values between 0 - 255. The radargram is
# then saved for input into caffe
#
################################################################################

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots B-scan.', usage='post-process \
	data and save radargram or plot annotated figure of radargram')
parser.add_argument('outputfile', help='name of output file including path')
parser.add_argument('field', default='Ez',help='name of field to be plotted, i.e. Ex, Ey, Ez')
parser.add_argument('directory',help='directory in which to save radargrams')
parser.add_argument('figure',help='option argument to plot annotated figure')
args = parser.parse_args()

# assign commandline inputs to variables
file = args.outputfile
dest_path = args.directory
field = args.field
path = '/rxs/rx1'

# open and xonvert data to numpy array
f = h5py.File(file, 'r')
data = f[path + '/' + field]
data_array = np.array(data)

def post_process(array,dest_path):

	# calculate SVD
	U, sigma, V = np.linalg.svd(array)

	# pull out dominant eigen images
	eigen_image = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])

	# subtract from original b-scan
	b_scan = array - eigen_image

	# normlise radargram (0-255)
	maxVal = b_scan.max()
	minVal = b_scan.min()
	dynamic = maxVal-minVal
	rescaled = np.divide((b_scan-minVal),dynamic)*255

	# Plot B-scan image and save
	fig = plt.figure(num=file, figsize=(20, 10))
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(rescaled,cmap='gray',interpolation='spline16',aspect='auto')
	fig.savefig(dest_path+"/"+file.split("/")[-1]+".png")

def annotated_plot(array,dest_path):

	# figure params
	plt.rcParams['xtick.labelsize'] = 25
	plt.rcParams['ytick.labelsize'] = 25 

	# plot annoted figure
	fig = plt.figure(figsize=(20, 10), facecolor='w', edgecolor='w')
	plt.imshow(array, extent=[0, array.shape[1], array.shape[0]*f.attrs['dt'], 0],\
	           interpolation='bicubic', aspect='auto', cmap='gray')
	plt.title("B-scan",fontsize=35)
	plt.xlabel('Trace number',fontsize=30)
	plt.ylabel('Time [s]',fontsize=30)
	plt.grid()
	cb = plt.colorbar()
	cb.set_label('Field strength [V/m]',fontsize=30)
	fig.savefig(dest_path+'/'+"B-scan", dpi=None, format='png', bbox_inches='tight', pad_inches=0.1)

	print "figure saved to "+dest_path

# if commandline flag 'figure' then annotate else
if args.figure:
	annotated_plot(data_array,dest_path)
else:
	post_process(data_array,dest_path)
#!/usr/bin/env python

################################################################################
# 
# Script to post-proccess simulated GPR B-scan and save as greyscal image. Post
# processing technique is known as singular value decomposition where the 3
# most dominant eigenimages are filtered and subtracted from the original B-scan
#
################################################################################

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots B-scan.', usage='post-process data and save radargram or plot annotated figure of radargram')
parser.add_argument('outputfile', help='name of output file including path')
parser.add_argument('field', default='Ez',help='name of field to be plotted, i.e. Ex, Ey, Ez')
parser.add_argument('directory',help='directory in which to save radargrams')
parser.add_argument('figure',help='optional argument to plot annotated B-scan i.e. "y" or "n"',default="n")
args = parser.parse_args()

# assign commandline inputs to variables
file = args.outputfile
dest_path = args.directory
field = args.field
path = '/rxs/rx1'

# open raw B-scan file
f = h5py.File(file, 'r')
data = f[path + '/' + field]
data_array = np.asarray(data)


def post_process(array,dest_path):

	# mean subtraction
	mean_radar = np.mean(array,axis=1)
	array = array - mean_radar[:,np.newaxis]

	# calculate SVD
	U, sigma, V = np.linalg.svd(array)

	# pull out dominant eigen images
	eigen_image = np.matrix(U[:, :2]) * np.diag(sigma[:2]) * np.matrix(V[:2, :])

	# subtract from original b-scan
	b_scan = array - eigen_image

	# normlise radargram (0-255)
	maxVal = b_scan.max()
	minVal = b_scan.min()
	dynamic = maxVal-minVal
	rescaled = np.divide((b_scan-minVal),dynamic)*255

	# Plot B-scan image
	fig = plt.figure(num=file, figsize=(20, 10))
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(rescaled,cmap='gray',interpolation='spline16',aspect='auto')
	img_name = dest_path+"/"+file.split("/")[-1]+".png"
	fig.savefig(img_name)

	# convert to greyscale
	input_img = plt.imread(img_name)
	arr = np.asarray(input_img)
	b_scan = arr[:,:,0]

	greyscale = (((b_scan - b_scan.min()) / (b_scan.max() - b_scan.min())) * 255.9).astype(np.uint8)

	out_img = Image.fromarray(greyscale)

	out_img.save(img_name)

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

	print ("figure saved to "+dest_path)

# if commandline flag 'figure' then annotate else
annotate = str(args.figure)
if annotate == 'y':
	annotated_plot(data_array,dest_path)
elif annotate == 'n':
	post_process(data_array,dest_path)
else:
	raise ValueError('Incorrect flag')

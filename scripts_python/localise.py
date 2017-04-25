#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os
import imutils
import time
import sys
caffe_root = '/home/pwhc/skycap/deep-learning/caffe/'
sys.path.insert(0, caffe_root +'python')
import caffe
import matplotlib.pylab as plt

parser = argparse.ArgumentParser(description='localise mine signature')

parser.add_argument('-i', '--image', required=True, dest='radargram', action='store', help='radargram root dir')

parser.add_argument('-d', '--directory', required=True, dest='output_dir', action='store',help='directory to store radargram with localised mine sig')

args = parser.parse_args()

# directory to store scaled images with bounding boxes
os.mkdir(args.output_dir)

# load image 
image = cv2.imread(args.radargram)

# set path to model definition file, pretrained weights and image to be classified
model_deploy = '/home/pwhc/skycap/gprlearn/models/caffe_files/deploy.prototxt'
pre_trained = '/home/pwhc/skycap/Results/Fifth_Model/stage5/saved_models/snapshot_iter_140.caffemodel'

print "Loading caffe model: " + str(pre_trained.split('/')[-1])

# set caffe to cpu mode
caffe.set_mode_cpu()

# define convnet
net = caffe.Classifier(	model_deploy,       # defines the structure of the model
                		pre_trained,		# contains the trained weights
	                	raw_scale = 255)    # define scale of image


def sliding_window(image, stepSize, windowSize):

	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def updte_heatmap(heatmap,prediction,x,y,winW,winH):

	window = heatmap[y:y + winW, x:x + winH]

	idx = np.where(window > 0)

	if len(idx[0] > 0):
		window[idx] = np.divide(window[idx] + prediction, 2)

	idx = np.where(window == 0)

	window[idx] = prediction

	heatmap[y:y + winW, x:x + winH] = window

	return heatmap 

def localise(image,stepSize,winW,winH):

	print ""
	print "Localising mine signature with sliding windows ..."
	print ""

	heatmap = np.zeros([image.shape[0],image.shape[1]])

	# create lists for bouding box co-ordinates
	image_predictions = []
	image_posX = []
	image_posY = []

	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(image, stepSize, windowSize=(winW, winH)):

	    # convnet prediction on window
		prediction = net.predict([window],oversample=False)

		# save highest scoring windows and coordinates
		if prediction[0][0] > 0.5:

			heatmap = updte_heatmap(heatmap,prediction[0][1],x,y,winW,winH)

			image_predictions.append(prediction[0][1])
			image_posX.append(x)
			image_posY.append(y)

	return image, image_predictions, image_posX, image_posY,heatmap

def visualise(output_dir,image_name,image,preictions,posX,posY,winW,winH):

	# naming convention to scaled images
	name = image_name.split('/')[-1].split('.')[0]

	# pull out coordinates of 
	top_left_x = min(posX) 
	top_left_y = min(posY)
	bottom_right_x = max(posX)
	bottom_right_y = max(posY)

	# visualise bounding boxes for image
	for i in range(len(posX)):
		cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x + winW, bottom_right_y + winH), (0, 255, 0),2)
	cv2.imshow("Window", image)
	cv2.waitKey(0)
	cv2.imwrite(output_dir+'/'+name+'.png',image)
	cv2.destroyAllWindows()

resised_images, predictions, posX, posY, heatmap = localise(image,64,128,128)

print heatmap.shape
print heatmap
plt.imshow(heatmap,format='jpg')

visualise(args.output_dir,args.radargram,image,predictions, posX, posY, 128, 128)
			
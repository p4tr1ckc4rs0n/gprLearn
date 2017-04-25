################################################################################
# loading_methods.py
#
# A number of possibly obsolete loading methods, most likely only used by the
# lasagne model.
################################################################################

import numpy as np
import os
from natsort import natsorted
from os import listdir
from os.path import isfile, join
from scipy.misc import imresize
from random import shuffle
import cv2

def create_y_data(fileid, col_num, dataDir):
    data = np.loadtxt(open(dataDir+"/mine"+fileid+".csv","rb"),delimiter=",")

    # translate mine positions to column positions
    dx = data[4]

    x_beg = data[0]
    x_end = data[2]

    col_beg = np.rint((x_beg/dx))-2
    if col_beg<0:
        col_beg=0   # this translates the mine pos to a columnn

    col_end = np.rint((x_end/dx))+2
    if col_beg > col_num-1:
        col_end = col_num-1

    # create y data
    y_data = np.zeros((1,col_num))
    y_data[:,col_beg-1:col_end] = 1

    return y_data

def read_mine_position(data_type,filename,col_num, dataDir):
    fileid = filename.replace("image","")
    # take in filename and check for mine file
    if os.path.isfile(dataDir+"/"+data_type+"/mine"+str(fileid)+".csv"):
        return create_y_data(fileid, col_num, dataDir+"/"+data_type), 1
    else:
        return np.zeros((1,col_num)), 0

def resize(data, (new_height, new_width)):
    if len(data.shape) > 2:
        new_array = np.zeros((new_height, new_width, data.shape[2]))
        for i in range(data.shape[2]):
            new_array[:,:,i] = cv2.resize(data[:,:,i], (new_height,new_width))

        return new_array
    else:
        return imresize(data,(new_height,new_width))

def load_data(data_type, dataDir, output_y_type, size):
    datasetSize = len([file for file in os.listdir(dataDir+"/"+data_type) if file.endswith(".csv") and "with" in file])
    count = 0
    dataset = []
    X = np.array([])
    Y = np.array([])

    filelist = [ f for f in listdir(dataDir+"/"+data_type) if isfile(join(dataDir+"/"+data_type,f)) ]

    shuffle(filelist)

    # read in X and Y training data
    for file in filelist:
        if file.endswith(".csv") and "with" in file:
            # read .csv data
            data = np.loadtxt(open(dataDir+"/"+data_type+"/"+file,"rb"),delimiter=",")

            if size != 0:
                data = resize(data, (size))

            if count == 0:
                #preallocate size of array
                X = np.empty([data.shape[0],data.shape[1],datasetSize])
                Y = np.empty([datasetSize])

            X[:,:,count] = data

            # if output_y_type=="windows":
            #     # if there is a mine, read in the y data, else create zeros
            #     output, boolean = read_mine_position(data_type, file.split(".")[0], data.shape[1], dataDir)
            #     Y[:,count] = output
            # elif output_y_type=="radargrams":
            #     output, boolean = read_mine_position(data_type, file.split(".")[0], data.shape[1], dataDir)
            #     Y[count] = boolean

            if "without" in file:
                Y[count] = 0
            else:
                Y[count] = 1

            count += 1

    return X, Y

def windows(input_x, input_y, window_width, size):
    number_images = input_x.shape[2]
    number_windows = input_x.shape[1] - window_width

    if size!= 0:
        all_windows_x = np.zeros((size[0],size[1],number_windows*number_images))
    else:
        all_windows_x = np.zeros((input_x.shape[0],window_width,number_windows*number_images))

    all_windows_y = np.zeros((window_width,number_windows*number_images))

    sliding_window_x = np.zeros((input_x.shape[0],window_width,number_images))
    sliding_window_y = np.zeros((window_width,number_images))


    for i in range(0,number_windows-1):
        sliding_window_x = input_x[:,i:window_width+i,:]
        sliding_window_y = input_y[i:window_width+i,:]

        if size != 0:
            all_windows_x[:,:,i*number_images:(i+1)*number_images]  = resize(sliding_window_x, (size))
        else:
            all_windows_x[:,:,i*number_images:(i+1)*number_images] = sliding_window_x

        all_windows_y[:,i*number_images:(i+1)*number_images] = sliding_window_y

    # adjust the y data to the correct shape
    y_data = np.zeros((1,number_windows*number_images))
    y_data[:,:] = np.amax(all_windows_y[:,:], axis=0)

    return all_windows_x, y_data.T

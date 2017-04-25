################################################################################
# model-caffe.py
#
# This file comprises the caffe model, including training and testing caffe nets
################################################################################

from __future__ import division
import numpy
import caffe
import argparse
import lmdb
import numpy as np
from os import listdir
from os.path import isfile, join
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import normalize
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import os, glob, sys
import cv2

# parse command-line arguments
parser = argparse.ArgumentParser(description='Process some input_images.')

parser.add_argument('-tr','--train', dest='trainDir', metavar='trainDir', type=str,
                   help='string for train directory', required=True)

parser.add_argument('-te','--test', dest='testDir', metavar='testDir', type=str,
                   help='string for test image names / class', required=True)

parser.add_argument('-v','--val', dest='valDir', metavar='valDir', type=str,
                   help='string for validation directory', required=True)

parser.add_argument('-c','--caffe', dest='caffeDir', metavar='dataDir', type=str,
                   help='string for caffe net directory', required=True)

parser.add_argument('-m','--meanimage', dest='meanimageDir', metavar='meanimageDir', type=str,
                   help='string for mean image dir', required=True)

parser.add_argument('-u','--uav', dest='uavDir', metavar='uavDIR', type=str,
                   help='string for UAV radargrams dir', required=True)

args = parser.parse_args()

# set database directories
trainDir = str(args.trainDir)
valDir = str(args.valDir)
testDir = str(args.testDir)
caffeDir = str(args.caffeDir)
meanimageDir = str(args.meanimageDir)
uavDir = str(args.uavDir)

def train(solver_txt):
    # train model normally

    caffe.set_mode_cpu()
    solver = caffe.SGDSolver(caffeDir+solver_txt)
    solver.net.forward()  # train net

    # not sure what this line is
    solver.test_nets[0].forward()

    ## train the net
    niter = 200 # number of iterations
    test_interval = 10 # test every n iterations
    train_loss = np.zeros(niter) # store losses
    test_acc = np.zeros(int(np.ceil(niter / test_interval))) # track accuracy
    output = np.zeros((niter, 1, 2)) # output

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['ip2'].data[:1]

        # run a full test every so often
        if it % test_interval == 0:
            print("Iteration ", str(it), " testing...")
            correct = 0
            total = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                               == solver.test_nets[0].blobs['label'].data)
                total += len(solver.test_nets[0].blobs['ip2'].data.argmax(1))

            test_acc[it // test_interval] = correct / total

    return niter, train_loss, test_interval, test_acc

def plot_loss(niter, train_loss, test_interval, test_acc):

    # plot the train loss and test accuracy
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.semilogy(arange(niter), train_loss, label="Train loss")
    ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r', label="Test Accuracy")
    ax1.set_xlabel('Iteration')
    ax1.set_xticks(np.arange(0,niter+10,20))
    ax1.grid()
    ax1.set_ylabel('Train loss')
    ax2.set_ylabel('Test accuracy')
    ax2.set_ylim([0,1])
    ax2.set_yticks(np.arange(0,1+0.1,0.1))
    ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))

    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels, loc='upper left')

    handles, labels = ax2.get_legend_handles_labels()
    lgd = ax2.legend(handles, labels, loc='lower left')
    plt.show()
    plt.savefig("results")

def load_model(modelDir):
    newest = max(glob.iglob(os.path.join(modelDir, '*.caffemodel')), key=os.path.getctime)
    return newest

def test_net(caffemodel, mean_bool):
    print "Loading from caffe model : " + str(caffemodel)

    # deifne mean image
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(meanimageDir+"/mean.binaryproto" , 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    mean = arr[0]

    # paths for caffe deploy and model files
    MODEL_FILE = caffeDir+"/deploy.prototxt"
    PRETRAINED = caffemodel

    print "CAFFE MODEL : "+str(PRETRAINED)

    # set caffe to cpu mode
    caffe.set_mode_cpu()

    # if bool == True load net WITH mean image subtraction
    if mean_bool:
        net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                               mean=mean,
                               raw_scale=255,
                               image_dims=(64, 64))
    else:
        net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                               raw_scale=255,
                               image_dims=(64, 64))

    return net

def val_images(net):
    print ""
    print "Classifying validation images..."

    input_images = [ f for f in listdir("/tmp/stage5/validation") if isfile(join("/tmp/stage5/validation",f)) ]

    mine_positive = np.zeros((1,len(input_images)))
    classes = np.zeros((1,len(input_images)))

    highest_image_name = ""

    count = 0

    print ""
    for i in range(len(input_images)):
        sys.stdout.write("\r%d%%" % np.round((count/len(input_images))*100))
        sys.stdout.flush()

        image_name = "/tmp/stage5/validation/"+str(input_images[i])
        image = caffe.io.load_image(image_name,False)

        prediction = net.predict([image], oversample=False)

        # save the actual class of the image
        if "without" in image_name:
            classes[0,i] = 0
        else:
            classes[0,i] = 1

        # save only the score given to the positive mine class
        mine_positive[0,i] = prediction[0][1]

        count += 1

    highest_image_idx = np.argmax(mine_positive[0])
    highest_image_name = "/tmp/stage5/validation/"+input_images[highest_image_idx]
    print highest_image_name

    return highest_image_name, classes, mine_positive

def graphics(classes, mine_positive, threshold):

    # calculate confusion matrix
    predictions = np.zeros((1,mine_positive.shape[1]))
    threshold = threshold

    predictions[mine_positive >= threshold] = 1
    predictions = predictions.astype(int)
    classes = classes.astype(int)

    classes = classes[0,:].T
    predictions = predictions[0,:].T
    mine_positive = mine_positive[0,:].T

    cf_matrix = confusion_matrix(classes, predictions)
    tp = cf_matrix[0,0]
    fp = cf_matrix[0,1]
    fn = cf_matrix[1,0]
    tn = cf_matrix[1,1]

    total_p = fn+tp
    total_f = tn+fp
    total = total_p+total_f

    TPR = tp/total_p
    FPR = fp/total_f
    PPV = tp/(tp+fp) 
    ACC = (tp+tn)/total

    print ""
    print "performance metrics at threshold %.2f:"%threshold
    print "tp rate : %.2f"%TPR
    print "fp rate : %.2f"%FPR
    print "ppv : %.2f"%PPV
    print "accuracy : %.2f"%ACC

    # plot confusion matrix
    plt.figure()
    plt.matshow(cf_matrix,cmap=plt.cm.gray_r)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(np.arange(0,2,1))
    plt.yticks(np.arange(0,2,1))
    plt.text(0,0,'True Positive: %d'%tn,va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    plt.text(0,1,'False Positive: %d'%fp,va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    plt.text(1,0,'False Negative: %d'%fn,va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    plt.text(1,1,'True Negative: %d'%tp,va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    plt.savefig("confusion matrix")

    # plot roc curve
    fpr, tpr, _ = roc_curve(classes, mine_positive)
    AUC = auc(fpr,tpr)

    plt.figure()
    plt.plot(fpr, tpr,label='ROC curve (area = %.2f)'%ACC)
    plt.plot([0, 1], [0, 1], 'k--',label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("roc curve")


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # set display defaults
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest' 
    plt.rcParams['image.cmap'] = 'gray'  
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    # make sure filters are plottable
    if len(data.shape) == 3:
        data = np.rollaxis(data,2,0)
        plt.imshow(data[0,:,:]); plt.axis('off')
        plt.show()
    else:
        plt.imshow(data); plt.axis('off')
        plt.show()

def visualise_filters(net, best_image):

    image = caffe.io.load_image(best_image,False)
    prediction = net.predict([image], oversample=False)

    print ""
    print "printing layer features and shapes : "
    for k, v in net.blobs.items():
        print str(k) + str(v.data.shape)

    print ""
    print "printing parameters and shapes : "
    for k, v in net.params.items():
        print str(k) + str(v[0].data.shape)
    print ""

    print "showing net filters and outputs ..."
    print ""

    # forward pass through network
    image = caffe.io.load_image(best_image,False)
    prediction = net.predict([image], oversample=False)

    # show parameters on first convolutional layer
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))

    # show activations of first convolutional layer
    feat = net.blobs['conv1'].data[0, :36]
    vis_square(feat)

    # show activations from first pooling layer
    feat = net.blobs['pool1'].data[0]
    vis_square(feat)

def uav_predictions(net,uav_dir):

    print ""
    print "UAV radargram class predictions :"

    images = [image for image in os.listdir(uav_dir)]

    for image in images:
        img = caffe.io.load_image(uav_dir+"/"+image,False)
        prediction = net.predict([img], oversample=False)

        print image,prediction

    print ""

if __name__ == '__main__':
    # train net (point toward solver file)
    niter, train_loss, test_interval, test_acc = train("/solver.prototxt")

    # show train / loss curve
    plot_loss(niter, train_loss, test_interval, test_acc)

    # load saved model (from /stage5/saved_models)
    caffemodel = load_model("/tmp/stage5/saved_models")

    # configure the net for testing, mean removal = True
    net_mean = test_net(caffemodel, True)

    # use model with test set images (not lmdb)
    highest_image_name, classes, mine_positive = val_images(net_mean)

    # confusion matrix and ROC curve
    graphics(classes, mine_positive,0.5)

    # visualise the filter kernels and layer activations
    visualise_filters(net_mean, highest_image_name)

    # point CNN at UAV data
    uav_predictions(net_mean, uavDir)
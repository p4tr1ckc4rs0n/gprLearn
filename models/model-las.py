################################################################################
# model-las.py
#
# This file comprises the Lasagne model
# This file only works using numpy data but does NOT train
################################################################################

from __future__ import division
import numpy as np
import os, sys, time
import cv2
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import gc
import lasagne
import theano
import theano.tensor as T
sys.dont_write_bytecode = True

#   Usage:
#
#     model.py -i <str>
#
#   where
#     --input <str> this is the directory in which input data files are stored
#     -i
#
#     --modelType <str> this is a string to define model type: "cnn" or "mlp"
#     -m
#

parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('-i','--input', dest='dataDir', metavar='dataDir', type=str,
                   help='string for data directory', required=True)

parser.add_argument('-m','--modelType', dest='modelType', metavar='modelType', type=str,
                   help='string for model type', required=True)

args = parser.parse_args()

dataDir = str(args.dataDir)
modelType = str(args.modelType)

#######################################################################
# 1: load up data
#######################################################################

print "Loading data ..."
X_train = np.load(dataDir+"/X_train.npy")
Y_train = np.load(dataDir+"/Y_train.npy")
X_test = np.load(dataDir+"/X_test.npy")
Y_test = np.load(dataDir+"/Y_test.npy")
Y_train = Y_train.T
Y_test = Y_test.T

######################################################
# 2: preprocessing
######################################################

# subtract mean image from every image
def zero_meaning(X_train, X_test):
    print "Mean subtraction ..."
    mean_image = np.mean(X_train,axis=2)

    return X_train - mean_image[:,:,np.newaxis], X_test - mean_image[:,:,np.newaxis], mean_image

X_train, X_test, mean = zero_meaning(X_train, X_test)

# normalise data to interval [0 1]
def normaliser(*arg):
    print "Normalising "+arg[1]+" data to range [0 1] ..."
    data = arg[0]
    if len(arg) > 2:
        min_val = arg[2]
        max_val = arg[3]
    else:
        min_val = np.amin(data)
        max_val = np.amax(data)

    return np.divide((data-min_val),(max_val-min_val)), min_val, max_val

X_train, min_val, max_val = normaliser(X_train, "train")
X_test, min_val, max_val = normaliser(X_test, "test", min_val, max_val)

# create negative of data? (just testing at present)
def negative(data):
    clean = np.ones((data.shape[0],data.shape[1]))
    return clean[:,:,np.newaxis] - data

# X_train = negative(X_train)
# X_test = negative(X_test)

######################################################
# 3: resize X data
######################################################

def resize(data, (new_height, new_width)):
    # if 3D array, resize each one individually, else just resize 2D array
    if len(data.shape) > 2:
        new_array = np.zeros((new_height, new_width, data.shape[2]))
        for i in range(data.shape[2]):
            new_array[:,:,i] = cv2.resize(data[:,:,i], (new_height,new_width), interpolation=3)

        return new_array
    else:
        return cv2.resize(data,(new_height,new_width))

new_size = (32,32)
X_train = resize(X_train, new_size)
X_test = resize(X_test, new_size)

print "Training X shape : "+str(X_train.shape)
print "Training Y shape : "+str(Y_train.shape)
print "Test X shape     : "+str(X_test.shape)
print "Test Y shape     : "+str(Y_test.shape)

print ""

######################################################
# 5: plot two examples
######################################################

# plots do not share a colorbar
def plot_example():
    print "plotting example ..."
    zero_idx = np.where(Y_train==0)[0][0]
    one_idx = np.where(Y_train==1)[0][0]

    plt.subplot(1, 2, 1)
    plt.imshow(X_train[:,:,zero_idx], cmap='gray')
    plt.title('Image for convnet, class = 0')

    plt.subplot(1, 2, 2)
    plt.imshow(X_train[:,:,one_idx], cmap='gray')
    plt.title("Image for convnet, class = 1")

    plt.show()

# plots share a colorbar
def plot_example2():
    print "plotting example ..."
    count = 0
    zero_idx = np.where(Y_train==0)[0][0]
    one_idx = np.where(Y_train==1)[0][0]
    vmin = np.amin(X_train)
    vmax = np.amax(X_train)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    for ax in axes.flat:
        if count == 0:
            im = ax.imshow(X_train[:,:,zero_idx], vmin=vmin, vmax=vmax, cmap='gray')
            ax.set_title('Image for convnet, class = 0')
            count += 1
        else:
            im = ax.imshow(X_train[:,:,one_idx], vmin=vmin, vmax=vmax, cmap='gray')
            ax.set_title('Image for convnet, class = 1')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

# plot example images
plot_example()

######################################################
# 5: pass images through Neural Network
######################################################
# set up some variables and then run

gc.collect()

num_epochs = 50
learning_rate = 0.00000001
batch_size = int(np.round(0.1*X_train.shape[2]))

im_height = X_train.shape[0]
im_width = X_train.shape[1]

print "Build model ..."

def build_mlp(input_var=None):
    # MLP

    network = lasagne.layers.InputLayer(shape=(None, 1, im_height, im_width),
                                     input_var=input_var)

    network = lasagne.layers.DropoutLayer(network, p=0.2)

    network = lasagne.layers.DenseLayer(
            network, num_units=600,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(
            network, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(
            network, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def build_cnn(input_var=None):
    # CONVNET

    # Input layer, size of images
    network = lasagne.layers.InputLayer(shape=(None, 1, im_height, im_width),
                                        input_var=input_var, regression=True)
    # dropout to avoid overfitting
    drop1 = lasagne.layers.DropoutLayer(network, p=0.2)

    # convolution layer
    conv = lasagne.layers.Conv2DLayer(
            drop1, num_filters=40, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.linear)

    # Max-pooling layer
    pool = lasagne.layers.MaxPool2DLayer(conv, pool_size=(2, 2), stride=2)

    # fully connected net
    dense = lasagne.layers.DenseLayer(pool, num_units=500,
        nonlinearity=lasagne.nonlinearities.linear)
    drop2 = lasagne.layers.DropoutLayer(dense, p=0.5)

    out = lasagne.layers.DenseLayer(drop2,num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def plotgraphs(training_error, validation_error, validation_accuracy):
    fig, ax1 = plt.subplots()

    line1, = ax1.plot(range(len(training_error)), training_error, 'r-', label = "training error")
    line2, = ax1.plot(range(len(validation_error)), validation_error, 'b-', label = "validation error")

    ax1.set_xlabel("Iteration number")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    line3, = ax2.plot(range(len(validation_accuracy)),validation_accuracy,'g-', label = "validation accuracy")
    ax2.set_ylabel("Percentage accuracy")
    ax2.set_ylim([0,1])

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')

    plt.show()

def run_net(X_train, y_train, X_test, y_test, model=modelType, num_epochs=num_epochs):
    # Load the dataset
    training_error = []
    validation_error = []
    validation_accuracy = []
    number_train_images = X_train.shape[2]

    # split validation set off (last 25% of dataset)
    X_val = X_train[:,:,np.round(number_train_images*0.75):number_train_images]
    y_val = y_train[np.round(number_train_images*0.75):number_train_images]

    y_train = y_train[0:number_train_images*0.75-1]
    X_train = X_train[:,:,0:number_train_images*0.75-1]

    # now reshape the data in to 4D tensors
    X_train = X_train.reshape((-1, 1, im_height, im_width))
    X_val = X_val.reshape((-1, 1, im_height, im_width))
    X_test = X_test.reshape((-1, 1, im_height, im_width))

    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    print ""
    print "Number of images with class = 1 : "
    print "     - Train : "+str(len(y_train[y_train==1])) +"/"+str(len(y_train))
    print "     - Val   : "+str(len(y_val[y_val==1])) +"/"+str(len(y_val))
    print "     - Test  : "+str(len(y_test[y_test==1])) +"/"+str(len(y_test))
    print ""

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)

    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])

    # compile a third function predicting output classification
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

            training_size = len(y_train)/batch_size
            sys.stdout.write("\r%d%%" % np.round((train_batches/training_size)*100))
            sys.stdout.flush()

        print ""

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc, pred = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        training_error.append(train_err / train_batches)
        validation_error.append(val_err / val_batches)
        validation_accuracy.append(val_acc / val_batches)

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0

    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc, pred = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    print ""

    print "Confusion matrix ..."
    # make predictions on test set
    y_predictions = predict_fn(X_test)

    print confusion_matrix(y_test, y_predictions)

    plotgraphs(training_error, validation_error, validation_accuracy)

# run the net
run_net(X_train, Y_train, X_test, Y_test)

gc.collect()
#

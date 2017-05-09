import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.metrics import confusion_matrix
import itertools
import glob


# Tensorflow saving and loading of models
def saveModel(sess,model_file):
    print("Saving model at: " + model_file)
    saver = tf.train.Saver()
    saver.save(sess, model_file)
    print("Saved")


def loadModel(sess, model_file):

    print("Loading model from: " + model_file)
    # load saved session
    loader = tf.train.Saver()
    loader.restore(sess, model_file)
    print("Loaded")



def get_image_data():
    print('Retrieving Data')


    X = []
    Y = []
    first = True
    for line in open('fer2013.csv'):
        #ingore first line of the file
        if first:
            first = False
        else:
            row = line.split(',')
            X.append([int(p) for p in row[1].split()])
            Y.append(int(row[0]))
            

    X, Y = np.array(X) / 255.0, np.array(Y)

    #if balance_one:
    # balance the 1 class
    X0, Y0 = X[Y!=1, :], Y[Y!=1]
    X1 = X[Y==1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.concatenate((Y0, [1]*len(X1)))

    #X, Y = get_data()
    print('Data Retrieved')
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y








def one_hot_label(y):
    N = len(y) #number of image labels
    num_labels = len(set(y)) # total set of labels
    one_hot = np.zeros((N, num_labels))
    for i in range(N):
        one_hot[i, y[i]] = 1
    return one_hot


def rgb2gray(img_rgb):
    ''' Function that converts RGB image to Grayscale'''
    img_grayscale = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
    return img_grayscale

def change_img_size(img,x_size,y_size):
    ''' Function that is used to change the size of an image '''
    img_new = misc.imresize(img.astype('uint8'), [x_size, y_size])
    return img_new




import pandas as pd
import numpy as np
import random

def training(train_data, label, epoch):
    """ 
    train_data: trainning data. Each row represents a sample. nxm matrix.
    label: labels for each sample. n dimension vector.
    epoch: an integer.
    Return a trained weight matrix.
    """

    # convert labels into integer
    classAll = np.unique(label).tolist()
    for i in range(0, len(classAll)):
        label[label == classAll[i]] = i

    no_feature = len(train_data[0])   # number of features

    k = 0
    w = np.zeros((len(classAll), no_feature))
    averaged_w = np.zeros((len(classAll), no_feature))
    s = np.zeros((len(classAll), 0))

    # averaged perceptron
    for t in range(0, epoch):
        for x in range(0, len(train_data)):
            # print("k = ", k)
            s = w.dot(train_data[x,:].T)  # calculate score for each class
            predict = np.argmax(s)  # return class with highest score
            # if predict correctly
            if predict == label[x]:
                averaged_w = averaged_w + w
                # print("no update")
            if predict != label[x]:  # if predict wrong
                w[label[x]] = w[label[x]] + train_data[x,:]
                w[predict] = w[predict] - train_data[x,:]
                averaged_w = averaged_w + w
            k += 1
    averaged_w = averaged_w / k
    return averaged_w


def predict(test_data, labelt, w):
    """ 
    test_data: testing data. Each row represents a sample. nxm matrix.
    label: labels for each test sample. n dimension vector.
    w: trained weight matrix.
    Return prediction for each test sample. 
    """

    classscore = np.zeros((len(labelt), 0))
    pre_y = []

    for x in range(0, len(test_data)):
        classscore = w.dot(test_data[x,:].T)
        predict = np.argmax(classscore)
        pre_y.append(predict)

    return pre_y


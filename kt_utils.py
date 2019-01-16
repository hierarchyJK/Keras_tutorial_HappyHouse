# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:JACK
@file:.py
@ide:untitled3
@time:2019-01-16 13:18:28
@month:一月
"""
import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
def mean_pred(y_true,y_pred):
    return K.mean(y_pred)
def load_dataset():
    train_dataset = h5py.File('F:\\吴恩达DL作业\\课后作业\\代码作业\\第四课第二周编程作业\\assignment\\KerasTutorial\\datasets\\train_happy.h5')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  ### your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  ### your train ser labels

    test_dataset = h5py.File('F:\\吴恩达DL作业\\课后作业\\代码作业\\第四课第二周编程作业\\assignment\\KerasTutorial\\datasets\\test_happy.h5')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  ### your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  ### your test set labels

    classes = np.array(test_dataset["list_classes"][:])   ### the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

import tensorflow as tf

from keras import regularizers

from keras.layers import Conv2D,Input,Concatenate,Conv2DTranspose,Multiply,Lambda,Activation,MaxPooling2D
from keras.layers import Maximum,Minimum,Average,Add,Flatten,Dense

from keras.models import Model
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt

def ssim_loss(y_true,y_pred):
    return ( 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,1.0)))


def dice_coef(y_true,y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = 2* intersection  
    score1=(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) )
    result=score/score1
    return 1-result
def IOU(y_true,y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    total=K.sum(y_pred_f)+K.sum(y_true_f)
    union=total-intersection
    IoU=(intersection+1)/(union+1)
    return 1-IoU
mse=tf.keras.losses.MeanSquaredError()
acc=tf.keras.metrics.Accuracy()
pre=tf.keras.metrics.Precision()
rec=tf.keras.metrics.Recall()
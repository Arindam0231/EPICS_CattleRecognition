import os
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras.backend as K

from keras.layers import Conv2D,Input,Concatenate,Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import regularizers
import Loss_Functions as L

class UNET1:
    def get_Model(self,height, width,channel,sn):
        input=Input((height,width,channel))
        conv1=Conv2D(sn*1,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(input)
        conv1=Conv2D(sn*1,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv1)
        maxpool1=MaxPooling2D()(conv1)

        conv2=Conv2D(sn*2,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(maxpool1)
        conv2=Conv2D(sn*2,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv2)
        maxpool2=MaxPooling2D()(conv2)



        conv3=Conv2D(sn*4,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(maxpool2)
        conv3=Conv2D(sn*4,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv3)
        maxpool3=MaxPooling2D()(conv3)

        conv4=Conv2D(sn*8,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(maxpool3)
        conv4=Conv2D(sn*8,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv4)
        maxpool4=MaxPooling2D()(conv4)

        conv5=Conv2D(sn*16,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(maxpool4)
        conv5=Conv2D(sn*16,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv5)
        maxpool5=MaxPooling2D()(conv5)

        convX=Conv2D(sn*32,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(maxpool5)
        convX=Conv2D(sn*32,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(convX)

        deconv0=Conv2DTranspose(sn*16,3,2,padding="same",activation="relu")(convX)
        concat0=Concatenate(axis=-1)([deconv0,conv5])
        conv0A=Conv2D(sn*16,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(concat0)
        conv0B=Conv2D(sn*16,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(conv0A)
        
        deconv1=Conv2DTranspose(sn*8,3,2,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv0B)
        concat1=Concatenate(axis=-1)([deconv1,conv4])
        conv6=Conv2D(sn*8,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(concat1)
        conv7=Conv2D(sn*8,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv6)

        deconv2=Conv2DTranspose(sn*4,3,2,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv7)
        concat2=Concatenate(axis=-1)([deconv2,conv3])
        conv8=Conv2D(sn*4,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(concat2)
        conv9=Conv2D(sn*4,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv8)

        deconv3=Conv2DTranspose(sn*2,3,2,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv9)
        concat3=Concatenate(axis=-1)([deconv3,conv2])
        conv10=Conv2D(sn*2,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(concat3)
        conv11=Conv2D(sn*2,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv10)

        deconv4=Conv2DTranspose(sn*1,3,2,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv11)
        concat4=Concatenate(axis=-1)([deconv4,conv1])
        conv12=Conv2D(sn*1,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(concat4)
        conv13=Conv2D(sn*1,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv12)

        output=Conv2D(channel,3,padding="same",activation="relu", kernel_regularizer = regularizers.l2(0.01))(conv13)

        model=Model(input,output)
        return model

    def __init__(self,height,width,channel,sn):
        self.model=self.get_Model(height,width,channel,sn)
        self.model.compile(optimizer="adam",loss=["mse"],metrics=["accuracy","Precision","Recall",L.ssim_loss,L.IOU,L.dice_coef])
       
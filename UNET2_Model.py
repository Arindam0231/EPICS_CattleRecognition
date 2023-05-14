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

import Loss_Functions as L


class UNET2:
    def Intermediate(self,x):
        a=tf.expand_dims(K.mean(x,axis=-1),axis=-1)
        b=tf.expand_dims(K.max(x,axis=-1),axis=-1)
        output=Concatenate(axis=-1)([a,b])
        return output
    def get_Model(self,height,width,channel,sn):
        input=Input((height,width,channel))
        conv1=Conv2D(sn*1,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(input)
        conv1=Conv2D(sn*1,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(conv1)
        maxpool1=MaxPooling2D()(conv1)

        conv2=Conv2D(sn*2,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(maxpool1)
        conv2=Conv2D(sn*2,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(conv2)
        maxpool2=MaxPooling2D()(conv2)

        
        conv3=Conv2D(sn*4,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(maxpool2)
        conv3=Conv2D(sn*4,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(conv3)
        maxpool3=MaxPooling2D()(conv3)

        conv4=Conv2D(sn*8,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(maxpool3)
        conv4=Conv2D(sn*8,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(conv4)
        maxpool4=MaxPooling2D()(conv4)

        conv5=Conv2D(sn*16,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(maxpool4)
        conv5=Conv2D(sn*16,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(conv5)
        maxpool5=MaxPooling2D()(conv5)

        conv6=Conv2D(sn*32,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(maxpool5)
        conv6=Conv2D(sn*32,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(conv6)

        deconv5=Conv2DTranspose(sn*16,3,2,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(conv6)
        lambda5=Lambda(self.Intermediate)(conv5)
        convL5=Conv2D(1,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(lambda5)
        add5=Multiply()([convL5,deconv5])

        deconv4=Conv2DTranspose(sn*8,3,2,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(add5)
        lambda4=Lambda(self.Intermediate)(conv4)
        convL4=Conv2D(1,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(lambda4)
        add4=Multiply()([convL4,deconv4])

        deconv3=Conv2DTranspose(sn*4,3,2,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(add4)
        lambda3=Lambda(self.Intermediate)(conv3)
        convL3=Conv2D(1,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(lambda3)
        add3=Multiply()([convL3,deconv3])

        deconv2=Conv2DTranspose(sn*2,3,2,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(add3)
        lambda2=Lambda(self.Intermediate)(conv2)
        convL2=Conv2D(1,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(lambda2)
        add2=Multiply()([convL2,deconv2])

        deconv1=Conv2DTranspose(sn*1,3,2,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(add2)
        lambda1=Lambda(self.Intermediate)(conv1)
        convL1=Conv2D(1,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(lambda1)
        add1=Multiply()([convL1,deconv1])


        output=Conv2D(channel,3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(0.01))(add1)

        model=Model(input,output)
        return model


    def __init__(self,height,width,channel,sn):
        self.model=self.get_Model(height,width,channel,sn)
        self.model.compile(optimizer="adam",loss=["mse"],metrics=["accuracy","Precision","Recall",L.ssim_loss,L.IOU,L.dice_coef])
       
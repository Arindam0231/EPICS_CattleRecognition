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
import Encoder_Model
import ImageSegmentation_Main 



test_image_path=os.path.join(os.getcwd(),"Dataset\Test")
WEIGHTS_PATH=os.path.join(os.getcwd(),"Weights")

class ExtractEncodings:

    def __init__(self,model):
        self.model=model

    def evaluate(self):
        TestingMetrics=pd.DataFrame(columns=["MSE","SSIM_LOSS","IOU_LOSS","ACCURACY","DICE_LOSS","PRECISION","RECALL"])
        x_test,y_test,name=self.getTestingdata()
        y_pred=self.model.predict(x_test,verbose=0,batch_size=2)
        y_predA=np.asarray(y_pred,dtype=np.uint8)
        loss=L.mse(y_test,y_pred).numpy()
        ssim=L.ssim_loss(y_test,y_predA).numpy()
        iou=L.IOU(y_test,y_predA).numpy()
        ac1=L.acc(y_test,y_pred).numpy()
        dice=L.dice_coef(y_test,y_predA).numpy()
        pr=L.pre(y_test,y_pred).numpy()
        re=L.rec(y_test,y_pred).numpy()
        metric_list=pd.DataFrame([[loss,ssim,iou,ac1,dice,pr,re]],columns=["MSE","SSIM_LOSS","IOU_LOSS","ACCURACY","DICE_LOSS","PRECISION","RECALL"])
        TestingMetrics=pd.concat([TestingMetrics,metric_list])
        print(TestingMetrics)

    def getTestingdata(self):
        x_test=[]
        y_test=[]
        name=[]
        dir_list=os.listdir(os.path.join(test_image_path,"Images"))
        for i in dir_list:
            img_path=os.path.join(test_image_path,"Images",i)
            msk_path=os.path.join(test_image_path,"Masks",i[:-4]+".png")
            img=cv2.imread(img_path,cv2.IMREAD_ANYDEPTH)
            mask=cv2.imread(msk_path,cv2.IMREAD_ANYDEPTH)
            gt=cv2.bitwise_and(img,mask)
            tempimg=np.asarray(img)
            tempimg=np.expand_dims(tempimg,axis=-1)
            tempimg=np.expand_dims(tempimg,axis=0)

            Predicter=ImageSegmentation_Main.Predict("UNET2").model
            tempimg=Predicter(tempimg)
            tempimg=np.squeeze(tempimg,axis=0)
            tempimg=np.squeeze(tempimg,axis=-1)
            tempimg=np.uint8(tempimg)

            pred_gt=np.bitwise_and(img,tempimg)
            x_test.append(pred_gt)
            y_test.append(gt)
            name.append(i[:-4])
       
        x=np.asarray(x_test)
        y=np.asarray(y_test)
        x=np.expand_dims(x,axis=-1)
        y=np.expand_dims(y,axis=-1)
        name=np.asarray(name)
        print(x.shape,y.shape)
        return x,y,name
    
    def getEncodings(self):
        x_test,y_test,name=self.getTestingdata()
        Intermediate_Model=Model(self.model.input,self.model.get_layer(name="features").output)
        img=x_test[np.random.randint(0,name.shape[0]-1)]
        img=np.expand_dims(img,axis=0)
        result=Intermediate_Model(img)
        result=K.flatten(result).numpy()
        

        forall=Intermediate_Model.predict(x_test)
        Encoded_Data=pd.DataFrame(columns=["Name","Encodings"])

        for i in range(0,len(forall)):
            row=pd.DataFrame([[name[i],K.flatten(forall[i]).numpy()]],columns=["Name","Encodings"])
            Encoded_Data=pd.concat([Encoded_Data,row],ignore_index=True)
        print(Encoded_Data.head(5))
        print(Encoded_Data.iloc[0,1].shape)

                
       
     

if __name__=="__main__":
    model=Encoder_Model.Encoder(512,512,1,16).model
    print(model.summary())
    model.load_weights(os.path.join(WEIGHTS_PATH,"Encoder_Model.h5"))
    Extractor=ExtractEncodings(model)
    Extractor.getEncodings()
    Extractor.evaluate()
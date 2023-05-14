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
import UNET2_Model 
import UNET1_Model
import Loss_Functions as L

test_image_path=os.path.join(os.getcwd(),"Dataset\Test")
WEIGHTS_PATH=os.path.join(os.getcwd(),"Weights")
class Predict:
    def __init__(self,model_name):
        if(model_name=="UNET2"):
            model=UNET2_Model.UNET2(512,512,1,16).model
            model.load_weights(os.path.join(WEIGHTS_PATH,"UNET2_2.h5"))
        if(model_name=="UNET1"):
            model=UNET1_Model.UNET1(512,512,1,16).model
            model.load_weights(os.path.join(WEIGHTS_PATH,"UNET1_2.h5"))
            
        self.model=model
    def predict(self,test_image_path):
        img=cv2.imread(test_image_path,cv2.IMREAD_ANYDEPTH)
        tempimg=np.asarray(img)
        tempimg=np.expand_dims(tempimg,axis=0)
        tempimg=np.expand_dims(tempimg,axis=-1)
        res=self.model.predict(tempimg,verbose=0)
        pred_mask=np.squeeze(res,axis=0)
        pred_mask=np.squeeze(pred_mask,axis=-1)
        cv2.imshow('Predicted Mask',pred_mask)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def get_random(self):
        dir_list=os.listdir(os.path.join(test_image_path,"Images"))
        curr_img=dir_list[np.random.randint(0,len(dir_list)-1)]
        img_path=os.path.join(test_image_path,"Images",curr_img)
        msk_path=os.path.join(test_image_path,"Masks",curr_img[:-4]+".png")
        mask=cv2.imread(msk_path,cv2.IMREAD_ANYDEPTH)
        cv2.imshow('Actual Mask',mask)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return img_path
    def getTestingData(self):
        x_test=[]
        y_test=[]
        dir_list=os.listdir(os.path.join(test_image_path,"Images"))
        for i in dir_list:
            img_path=os.path.join(test_image_path,"Images",i)
            msk_path=os.path.join(test_image_path,"Masks",i[:-4]+".png")
            img=cv2.imread(img_path,cv2.IMREAD_ANYDEPTH)
            mask=cv2.imread(msk_path,cv2.IMREAD_ANYDEPTH)
            x_test.append(img)
            y_test.append(mask)
        x=np.asarray(x_test)
        y=np.asarray(y_test)
        x=np.expand_dims(x,axis=-1)
        y=np.expand_dims(y,axis=-1)
        return x,y
    def evaluate(self):
        TestingMetrics=pd.DataFrame(columns=["MSE","SSIM_LOSS","IOU_LOSS","ACCURACY","DICE_LOSS","PRECISION","RECALL"])
        x_test,y_test=self.getTestingData()
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
        
# if __name__=="__main__":
#     model_name="UNET1"
    
#     Predicter=Predict(model_name)
#     print(Predicter.model.summary())
#     img_path=Predicter.get_random()
#     Predicter.predict(img_path)
#     Predicter.evaluate()
    

    

   

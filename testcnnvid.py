import cv2 as cv
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from sklearn.externals import joblib
from time import sleep
IMG_SIZE=100
LR=1e-3
convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')
convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,128,5,activation='relu')
##convnet=max_pool_2d(convnet,5)
##convnet=conv_2d(convnet,256,5,activation='relu')
##convnet=max_pool_2d(convnet,5)
##convnet=conv_2d(convnet,128,5,activation='relu')
##convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=fully_connected(convnet,4096,activation='relu')
convnet=dropout(convnet,0.8)
convnet=fully_connected(convnet,102,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')
model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=0)
model.load('/home/tharunsai/Documents/python/model_100_epoch_50_4096.tfl')
q=os.listdir('/home/tharunsai/Documents/python/101_ObjectCategories')
cap=cv.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('image',frame)
    img_data=cv.resize(gray,(IMG_SIZE,IMG_SIZE))
    a=np.array(img_data)
    data=a.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out=model.predict([data])
    b=np.argmax(model_out)
    print(q[b])
    sleep(0.5)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()

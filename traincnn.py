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
IMG_SIZE=100
LR=1e-3
convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')
convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,128,5,activation='relu')
convnet=max_pool_2d(convnet,5)
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
##convnet=conv_2d(convnet,8,5,activation='relu')
##convnet=max_pool_2d(convnet,5)
##convnet=conv_2d(convnet,16,5,activation='relu')
##convnet=max_pool_2d(convnet,5)
##convnet=conv_2d(convnet,32,5,activation='relu')
##convnet=max_pool_2d(convnet,5)
####convnet=conv_2d(convnet,256,5,activation='relu')
####convnet=max_pool_2d(convnet,5)
####convnet=conv_2d(convnet,128,5,activation='relu')
####convnet=max_pool_2d(convnet,5)
##convnet=conv_2d(convnet,16,5,activation='relu')
##convnet=max_pool_2d(convnet,5)
##convnet=conv_2d(convnet,8,5,activation='relu')
##convnet=max_pool_2d(convnet,5)
##convnet=fully_connected(convnet,4096,activation='relu')
##convnet=dropout(convnet,0.8)
##convnet=fully_connected(convnet,102,activation='softmax')
##convnet=regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')
model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=0)
model.load('/home/tharunsai/Documents/python/model_100_epoch_50_4096.tfl')
x='/home/tharunsai/Documents/python/ss.jpg'
q=os.listdir('/home/tharunsai/Documents/python/101_ObjectCategories')
img_data=cv.imread(x,cv.IMREAD_GRAYSCALE)
plt.imshow(img_data)
plt.show()
img_data=cv.resize(img_data,(IMG_SIZE,IMG_SIZE))
a=np.array(img_data)
data=a.reshape(IMG_SIZE,IMG_SIZE,1)
model_out=model.predict([data])
for i in range(0,3):
    b=np.argmax(model_out)
    print(q[b],end=',')
    model_out[0][b]=0
##for i in os.listdir(x):
##    w=os.path.join(x,i)
##    img_data=cv.imread(w,cv.IMREAD_GRAYSCALE)
##    img_data=cv.resize(img_data,(IMG_SIZE,IMG_SIZE))
##    a=np.array(img_data)
##    data=a.reshape(IMG_SIZE,IMG_SIZE,1)
##    model_out=model.predict([data])
##    b=np.argmax(model_out)
##    print(q[b])
##    for j in range(0,3):
##        b=np.argmax(model_out)
##        print(q[b],end=',')
##        model_out[0][b]=0
##    print('\n')
##def create_test_data():
##    test_data=[]
##    for doc in (os.listdir(file)):
##        path1=os.path.join(file,doc)
##        path=os.path.join(path1,os.listdir(path1)[0])
##        img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
##        img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
##        test_data.append([np.array(img_data),create_lable(doc)])
##    shuffle(test_data)
##    np.save('/home/tharunsai/Documents/python/test_data.npy',test_data)
##    return test_data
##testing=create_test_data()
##for data,num in testing:
##    img=data.reshape(50,40,1)
##model_out=model.predict([img])

import cv2
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
file='/home/tharunsai/Documents/python/101_ObjectCategories/'
x=os.listdir('/home/tharunsai/Documents/python/101_ObjectCategories/')
IMG_SIZE=50
LR=1e-3
MODEL_NAME='CALTECH_101'
def create_lable(image_name):
    z=np.zeros(102,dtype='int32')
    a=0
    #x=os.listdir('/home/tharunsai/Documents/python/101_ObjectCategories/')
    for i in range(0,len(x)):
        if image_name==x[i]:
            a=i
            break
    z[a]=1
    return z
def create_train_data():
    training_data=[]
    for doc in os.listdir(file):
        path1=os.path.join(file,doc)
        for img in os.listdir(path1):
            path=os.path.join(path1,img)
            img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img_data),create_lable(doc)])
    shuffle(training_data)
    np.save('/home/tharunsai/Documents/python/train_data.npy',training_data)
    return training_data
def create_test_data():
    test_data=[]
    for doc in (os.listdir(file)):
        path1=os.path.join(file,doc)
        path=os.path.join(path1,os.listdir(path1)[0])
        img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
        test_data.append([np.array(img_data),create_lable(doc)])
    shuffle(test_data)
    np.save('/home/tharunsai/Documents/python/test_data.npy',test_data)
    return test_data
train=create_train_data()
test=create_test_data()
x_train=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_train=[i[1] for i in train]
x_test=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_test=[i[1] for i in test]
tf.reset_default_graph()
convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')
convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,128,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=fully_connected(convnet,2,activation='relu')
convnet=dropout(convnet,0.8)
convnet=fully_connected(convnet,102,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')
model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=0)
print(type(model))
#model.fit({'input':x_train},{'targets':y_train},n_epoch=10,validation_set=({'input':x_test},{'targets':y_test}),snapshot_step=500,show_metric=True,run_id=MODEL_NAME)
model.fit(x_train,y_train,n_epoch=20,validation_set=(x_test,y_test),snapshot_step=500,show_metric=True,run_id=MODEL_NAME)
#joblib.dump(model,'/home/tharunsai/Documents/python/cnn.plk')
##saver=tf.train.Saver()
##save_path=saver.save(model,'/home/tharunsai/Documents/python/cnn.plk')
##print(save_path)
model.save('/home/tharunsai/Documents/python/model.tfl')

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 21:27:15 2017
@author: Administrator
"""
import os
import shutil
import cv2
import h5py
import numpy as np
import pandas as pd
model_path=model_dogs_Xception.h5
model=load_model(model_path)
test_filenames=os.listdir(test_data_dir)
test_img=[]
predictions=[]
for file_path in test_filenames:
    img=cv2.imread(test_data_dir+file_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(299,299),interpolation=cv2.INTER_CUBIC)
    test_img.append(img)
    test_img=np.array(test_img)
    test_img = test_img.astype('float32')
    test_img/=255
    pre=model.predict(test_img,batch_size=1)[0]
    predictions.append(pre)
    test_img=[]
    
probs=np.array(predictions)
classes=np.argmax(probs,1)
    
#clsses-labels
class_indices=np.load('class_indices.txt.npy')
#(int,int)
class_indices=class_indices.tolist()
#(int,str)
value_indices={v:k for k,v in class_indices.items()}
true_class=[]
for i in range(len(classes)):
   true_class.append(value_indices[classes[i]])

with open('submit.txt','a') as fp:
    for i in range(len(test_filenames)):
        fp.write(str(true_class[i])+"\t"+str(test_filenames[i].split(".")[0])+'\n')
    fp.close
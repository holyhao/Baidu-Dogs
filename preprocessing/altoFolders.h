# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:15:14 2017
@author: holy
"""
import os
import shutil
import pandas as pd
import numpy as np
label = pd.read_csv("train.txt",sep='\s+',encoding='utf-8',escapechar='\n')
#label=np.loadtxt("data_train_image.txt",delimiter='\t')
train_filenames=os.listdir('train')
def ex_mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname) 
def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)      
ex_mkdir('train2')
for iter in label.index:
    name=label.iloc[iter,0]
    i=label.iloc[iter,1]
    ex_mkdir('train2/'+str(i))
    shutil.copy('train/'+name+'.jpg', 'train2/'+str(i)+'/'+name+'.jpg')
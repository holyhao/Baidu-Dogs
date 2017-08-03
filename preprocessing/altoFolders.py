# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:15:14 2017
@author: holy
"""
import os
import shutil
import pandas as pd
import numpy as np
label = pd.read_csv("originaldata.txt",sep='\s+',encoding='utf-8',escapechar='\n')
train_filenames=os.listdir('originaldata')
def ex_mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname) 
def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)      
ex_mkdir('../data/train')
for iter in label.index:
    name=label.iloc[iter,0]
    i=label.iloc[iter,1]
    ex_mkdir('train2/'+str(i))
    shutil.copy('originaldata'+name+'.jpg', '../data/train/'+str(i)+'/'+name+'.jpg')
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:15:14 2017
@author: holy
"""
filename ='../data/train'
train_dir='../data/train'
val_dir='../data/val'
ls=os.listdir(filename)
def ex_mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname) 
def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)      
ex_mkdir(val_dir)
for i in range(0,len(ls)):
    ex_mkdir(val_dir+str(ls[i]))
    data=os.listdir(train_dir+str(ls[i]))
    for j in range(0,int(0.2*len(data))):#%20 for validation
        name=data[j]
        shutil.move(train_dir+str(ls[i])+'/'+name,val_dir+str(ls[i])+'/'+name)
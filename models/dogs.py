#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:49:09 2017
@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import h5py
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Input,GlobalAveragePooling2D
from keras.utils import vis_utils,plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from vgg19 import VGG19


#模型的构建
img_rows, img_cols, img_channel = 400, 400, 3
base_model = VGG19(weights='imagenet', include_top=False,input_shape=(img_rows, img_cols, img_channel))
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(1024, activation='relu'))
add_model.add(Dense(100, activation='softmax'))
model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
#冻结某些层设置
#for layer in model.layers[:85]:
 #   layer.trainable = False
#打印网络结构
#model.summary()
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
#参数设置
batch_size = 32
epochs = 50
train_data_dir="data/train"
val_data_dir="data/val"

#plot_model(model,to_file='model.png')
train_datagen = ImageDataGenerator(
        rotation_range=30, 
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
		
val_datagen=ImageDataGenerator(rescale=1./255)
		
val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_s,
        class_mode='categorical')

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
#保存映射表
#with h5py.File("class_indices.h5") as h:
    #h.create_dataset("class_indices",data=train_generator.class_indices)
#np.save('class_indices.txt', train_generator.class_indices)

history = model.fit_generator(
     train_generator,
     steps_per_epoch=train_generator.samples/batch_size,
     epochs=epochs,
	 validation_data=val_generator,
     validation_steps=batch_s
     #callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
 )
model.save('model_dogs_VGG19_400*400_full.h5') 


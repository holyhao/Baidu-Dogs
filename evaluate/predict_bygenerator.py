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
model_path="../models/model_dogs_Xception.h5"
test_data_dir="../data/test"
val_data_dir="../data/val"
model=load_model(model_path)
test_datagen = ImageDataGenerator(rescale=1./255)
batch_size=64
#for generate class_indices
val_generator = test_datagen.flow_from_directory(
    val_data_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(299, 299),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

y= model.predict_generator(test_generator, test_generator.samples/batch_size + 1)
y_max_idx = np.argmax(y1, 1)
predict_path = 'submission.txt'

with open(predict_path,'a') as fp:
    for i, idx in enumerate(y_max_idx):
        fp.write(str(label_idxs[idx][0]) + '\t' + test_generator.filenames[i][6:-4] + '\n')
    fp.close
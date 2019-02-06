# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:46:33 2016

@author: oyu
"""
import numpy as np
from dataset.easymult import MyDataset


#
data_dir = "C:/Users/waka-lab/Documents/data/data/easymult/"
train_data = MyDataset(data_dir, "train")
img_sum = np.zeros((3, 256, 256))
length = len(train_data)
for i in range(length):

    data, l = train_data.get_example(i)
    img_sum += data
mean = img_sum / length
mean = mean.sum(axis=1) / 255
mean = mean.sum(axis=1) / 255
print(mean)
print(mean * 255)

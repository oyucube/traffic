# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:46:33 2016

@author: oyu
"""
import os
# os.environ["CHAINER_TYPE_CHECK"] = "0" #ここでオフに  オンにしたかったら1にするかコメントアウト
import numpy as np
# 乱数のシード固定
#
# i = np.random()
# np.random.seed()

from sklearn.datasets import fetch_mldata
import pickle
from dataset.libtraindata import make_train_data
from PIL import Image

# 教師データの読み込み
mnist = fetch_mldata('MNIST original', data_home=".")
with open('dataset/m09.pickle', 'rb') as f:
    dic = pickle.load(f)
data = dic["data"]
num_class = dic["num_class"]
# target_c = dic["target_combinations"]
target_c = ""
train_data, train_target = make_train_data(data, mnist.data, num_class, size=112)


datadir = "C:/Users/waka-lab/Documents/data/data/"

# output dir
savedir = datadir + "mnist45/"
f_log = open(savedir + 'log.txt', 'w')
f_label = open(savedir + 'label.txt', 'w')

print(train_target[0])
print(np.argmax(train_target[0]))
print(train_data[0].shape)
print(train_data[0])
# img = Image.fromarray(np.uint8(train_data[0]))
# img.save(savedir + str(0) + ".jpg")

data_max = len(train_data)
for i in range(data_max):
    img = Image.fromarray(np.uint8(train_data[i]))
    img.save(savedir + str(i) + ".jpg")
    f_label.write(str(i) + ".jpg, " + str(np.argmax(train_target[i])) + "\n")
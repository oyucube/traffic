# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:05:52 2016

@author: oyu
"""

from sklearn.datasets import fetch_mldata
import numpy as np
import random
from itertools import combinations
from PIL import Image
import math

def afin(a, b, x):
    k = b/a
    r = np.empty((a, a))

    for i in range(0, a):
        for j in range(0, a):
            ax = int(k*i)
            ay = int(k*j)
            r[i][j] = x[ax][ay]
    return r


def main():
    mnist = fetch_mldata('MNIST original', data_home=".")
    numin = 70000

    mnist_label_index = np.load("./mnist/label.npy")
    mnist_max_each_label = np.load("./mnist/count.npy")

# ここで出力を指定
    img_size = 112
    file_id = "s09"
    target_number = "0123456789"
    num_each_class = 3000
    num_each_class_test = 50

    min_size = 20
    max_size = 80

    # このディレクトリに画像、ラベルを保存
    datadir = "C:/Users/waka-lab/Documents/data/data/"
    savedir = datadir + file_id + "/"
    f_log = open(savedir + 'log.txt', 'w')
    f_test = open(savedir + 'test.txt', 'w')
    f_train = open(savedir + 'train.txt', 'w')

    # sizeの分布を調べる
#    count_size = np.zeros(100)

    # 目的の組み合わせを生成
    num_class = 10
    for i in range(num_class):
        # mnist における数字の数
        n1_max = mnist_max_each_label[i]

        for j in range(num_each_class + num_each_class_test):
            digit_id1 = random.randint(0, n1_max - 1)

            data_id = num_each_class * i + j
            while(True):
                size = int(min_size * math.exp(math.log(max_size / min_size) * random.random()))
#                 size = random.randint(min_size, max_size - 1)
                position = (1 - size / 112) * np.random.rand(2)

                black = np.zeros((img_size, img_size))
                mnist_id = mnist_label_index[i][digit_id1]
                img1 = mnist.data[mnist_id].reshape(28, 28)
                x = int(position[0] * img_size)
                y = int(position[1] * img_size)
                black[x:x + size, y:y + size] = afin(size, 28, img1)

                if j < num_each_class:
                    img = Image.fromarray(black).convert("L")
                    img.save(savedir + str(data_id) + ".jpg")
                    f_train.write(str(data_id) + ".jpg, " + str(i) + "\n")

                else:
                    test_id = j - num_each_class + num_each_class_test * i
                    img = Image.fromarray(black).convert("L")
                    img.save(savedir + "test" + str(test_id) + ".jpg")
                    f_test.write("test" + str(test_id) + ".jpg, " + str(i) + "\n")
                break


if __name__ == '__main__':
    main()

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
    file_id = "m09"
    target_number = "0123456789"
    num_each_class = 1000
    num_each_class_test = 50

    min_size = 20
    max_size = 80

    # このディレクトリに画像、ラベルを保存
    datadir = "C:/Users/waka-lab/Documents/data/data/"
    savedir = datadir + file_id + "/"
    f_log = open(savedir + 'log.txt', 'w')
    f_test = open(savedir + 'test.txt', 'w')
    # f_train = open(savedir + 'train.txt', 'w')

    # sizeの分布を調べる
#    count_size = np.zeros(100)

    # 目的の組み合わせを生成
    target_combinations = np.array((list(combinations(target_number, 2)))).astype(np.int32)
    num_class = target_combinations.shape[0]

    print("target {}\n num class:{}".format(target_combinations, num_class))

    for i in range(num_class):
        # mnist における数字の数
        n1_max = mnist_max_each_label[target_combinations[i][0]]
        n2_max = mnist_max_each_label[target_combinations[i][1]]

        for j in range(num_each_class + num_each_class_test):
            digit_id1 = random.randint(0, n1_max -1)
            digit_id2 = random.randint(0, n2_max -1)

            id = num_each_class * i + j
            while(True):
                size = np.random.randint(min_size, max_size, (2, 1)).astype(np.int32)
                position = (1 - size / 112) * np.random.rand(2, 2)
                center = position + size / 112 / 2

                test = np.abs(center[0] - center[1]) - np.sum(size / 112) / 2
                t = np.sum(np.sign(test))

                # 領域が被るか判定
                if t > -2:
                    black = np.zeros((img_size, img_size))
                    mnist_id = mnist_label_index[target_combinations[i][0]][digit_id1]
                    img1 = mnist.data[mnist_id].reshape(28, 28)
                    x = int(position[0][0] * img_size)
                    y = int(position[0][1] * img_size)
                    black[x:x + int(size[0]), y:y + int(size[0])] = afin(int(size[0]), 28, img1)

                    mnist_id = mnist_label_index[target_combinations[i][1]][digit_id1]
                    img2 = mnist.data[mnist_id].reshape(28, 28)
                    x = int(position[1][0] * img_size)
                    y = int(position[1][1] * img_size)
                    black[x:x + int(size[1]), y:y + int(size[1])] = afin(int(size[1]), 28, img2)

                    if j < num_each_class:
                        img = Image.fromarray(black).convert("L")
                        img.save(savedir + str(id) + ".jpg")
                        f_train.write(str(id) + ".jpg, " + str(i) + "\n")

                    else:
                        test_id = j - num_each_class + num_each_class_test * i
                        img = Image.fromarray(black).convert("L")
                        img.save(savedir + "test" + str(test_id) + ".jpg")
                        f_test.write("test" + str(test_id) + ".jpg, " + str(i) + "\n")
                    break


if __name__ == '__main__':
    main()

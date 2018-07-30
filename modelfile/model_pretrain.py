# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import make_sampled_image
from env import xp


class BASE(chainer.Chain):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, n_step=2, gpu_id=-1):
        super(BASE, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            # in 256 * 256 * 3
            #
            # 32 * 32 ->  16 * 16 -> 8 * 8 -> 1
            #         pool      pool      full
            cnn_1_1=L.Convolution2D(3, 32, 3, pad=1),
            cnn_1_2=L.Convolution2D(32, 32, 3, pad=1),
            cnn_2_1=L.Convolution2D(32, 64, 3, pad=1),
            cnn_2_2=L.Convolution2D(64, 64, 3, pad=1),
            cnn_3_1=L.Convolution2D(64, 64, 3, pad=1),
            cnn_3_2=L.Convolution2D(64, 64, 3, pad=1),
            # full_1=L.Linear(4 * 4 * 64, 256),
            full_2=L.Linear(None, 5),


            norm_1_1=L.BatchNormalization(32),
            norm_1_2=L.BatchNormalization(32),
            norm_2_1=L.BatchNormalization(64),
            norm_2_2=L.BatchNormalization(64),
            norm_3_1=L.BatchNormalization(64),
            norm_3_2=L.BatchNormalization(64),
            norm_f1=L.BatchNormalization(256),
        )

        #
        # img parameter
        #
        if gpu_id == 0:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.img_size = img_size
        self.gsize = 32
        self.train = True
        self.n_unit = n_units
        self.num_class = n_out

    def __call__(self, x, target, bf=False):
        y = self.glimpse_forward(x)
        if chainer.config.train:
            loss = F.softmax_cross_entropy(y, target)
            return loss
        else:
            y = F.softmax(y)
            index = xp.array(range(y.data.shape[0]))
            acc = y.data[index, target]
            acc = chainer.cuda.to_cpu(acc)
            return acc.sum()

    def glimpse_forward(self, x):
        h = F.relu(self.norm_1_1(self.cnn_1_1(x)))
        h = F.relu(self.norm_1_2(F.max_pooling_2d(self.cnn_1_2(h), 2, stride=2)))
        h = F.relu(self.norm_2_1(self.cnn_2_1(h)))
        h = F.relu(self.norm_2_2(F.max_pooling_2d(self.cnn_2_2(h), 2, stride=2)))
        h = F.relu(self.norm_3_1(self.cnn_3_1(h)))
        h = F.relu(self.norm_3_2(F.max_pooling_2d(self.cnn_3_2(h), 2, stride=2)))
        h = F.relu(self.norm_f1(self.full_1(h)))
        return h


class SAF(BASE):
    pass

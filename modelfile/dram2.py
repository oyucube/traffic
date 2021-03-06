# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda
import math
import numpy as np
import time
import make_sampled_image
from env import xp


class SAF(chainer.Chain):
    def __init__(self, n_units, n_out, img_size, var=0.18, gpu_id=-1):
        super(SAF, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            glimpse_cnn_1=L.Convolution2D(1, 8, 5),  # in 28 out 24
            glimpse_cnn_2=L.Convolution2D(8, 32, 5),  # in 24 out 20
            glimpse_cnn_3=L.Convolution2D(32, 64, 5),  # in 20 pool 10 out 6
            glimpse_full=L.Linear(8 * 8 * 64, n_units),
            glimpse_loc=L.Linear(2, n_units),

            # baseline network 強化学習の期待値を学習し、バイアスbとする
            baseline=L.Linear(n_units, 1),

            # 記憶を用いるLSTM部分
            rnn_1=L.LSTM(n_units, n_units),
            rnn_2=L.LSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(n_units, 2),

            # 入力画像を処理するネットワーク
            context_cnn_1=L.Convolution2D(1, 4, 5),  # 56 to 52 pooling: 26
            context_cnn_2=L.Convolution2D(4, 4, 5),  # 26 to 22 pooling
            context_cnn_3=L.Convolution2D(4, 4, 4),  # 22 to 16

            class_full=L.Linear(n_units, n_out)
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

        self.var = var
        self.vars = var
        self.n_unit = n_units
        self.num_class = n_out
        self.r = 0.5

    def reset(self):
        self.rnn_1.reset_state()
        self.rnn_2.reset_state()

    def my_name(self):
        return "deep recurrent attention model 2"

    def __call__(self, x, target, num_lm, batch_size=1, train=1, debug=0, n_step=1):

        if train == 1:
            self.reset()
            l, b1 = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm = self.make_img(x, l, num_lm, random=1)
                    l1, y, b = self.recurrent_forward(xm, lm)

                    loss = self.cul_loss(y, target, l, lm, num_lm, b1)
                    return loss / num_lm
                else:
                    xm, lm = self.make_img(x, l, num_lm, random=0)
                    l1, y, b = self.recurrent_forward(xm, lm)
                l = l1
                b1 = b

        elif train == 0:
            sum_accuracy = 0
            ydata = xp.zeros((num_lm, self.num_class))
            self.reset()
            l, b1 = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm = self.make_img(x, l, num_lm, random=0)
                    l1, y, b = self.recurrent_forward(xm, lm)

                    accuracy = y.data * target.data
                    sum_accuracy += xp.sum(accuracy)
                    ydata += y.data

                    return sum_accuracy / (num_lm * n_step), ydata / n_step, xp.sum(accuracy) / num_lm
                else:
                    xm, lm = self.make_img(x, l, num_lm, random=0)
                    l1, y, b = self.recurrent_forward(xm, lm)
                    accuracy = y.data * target.data
                    sum_accuracy += xp.sum(accuracy)
                    ydata += y.data
                l = l1
        elif train == 2:
            sum_accuracy = 0
            ydata = xp.zeros((num_lm, self.num_class))
            self.reset()
            l_list = xp.zeros((n_step, num_lm, 2))
            s_list = xp.zeros((n_step, num_lm, 1))
            l, b1 = self.first_forward(x, num_lm)
            l_list[0] = l.data
            s_list[0] = self.gsize / self.img_size
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm = self.make_img(x, l, num_lm, random=0)
                    l1, y, b = self.recurrent_forward(xm, lm)

                    accuracy = y.data * target.data
                    sum_accuracy += xp.sum(accuracy)
                    ydata += y.data
                    return sum_accuracy / (num_lm * n_step), ydata / n_step, xp.sum(accuracy) / num_lm, l_list, s_list
                else:
                    xm, lm = self.make_img(x, l, num_lm, random=0)
                    l1, y, b = self.recurrent_forward(xm, lm)
                    accuracy = y.data * target.data
                    sum_accuracy += xp.sum(accuracy)
                    ydata += y.data
                l = l1
                l_list[i + 1] = l.data
                s_list[i + 1] = self.gsize / self.img_size

            return False

    def first_forward(self, x, num_lm):
        self.rnn_1(Variable(xp.zeros((num_lm, self.n_unit)).astype(xp.float32)))
        h2 = F.relu(self.context_cnn_1(F.max_pooling_2d(x, 2, stride=2)))
        h3 = F.relu(self.context_cnn_2(F.max_pooling_2d(h2, 2, stride=2)))
        h4 = F.relu(self.context_cnn_3(F.max_pooling_2d(h3, 2, stride=2)))
        h5 = F.relu(self.rnn_2(h4))

        l = F.sigmoid(self.attention_loc(h5))
        b = F.sigmoid(self.baseline(Variable(h5.data)))
        return l, b

    def recurrent_forward(self, xm, lm):
        hgl = F.relu(self.glimpse_loc(lm))
        hg1 = F.relu(self.glimpse_cnn_1(Variable(xm)))
        hg2 = F.relu(self.glimpse_cnn_2(hg1))
        hg3 = F.relu(self.glimpse_cnn_3(F.max_pooling_2d(hg2, 2, stride=2)))
        hgf = F.relu(self.glimpse_full(hg3))

        hr1 = F.relu(self.rnn_1(hgl * hgf))
        # ベクトルの積
        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, y, b

    # loss 関数を計算

    def cul_loss(self, y, target, l, lm, num_lm, b):

        l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
        m1, m2 = F.split_axis(lm, indices_or_sections=2, axis=1)
        ln_p = ((l1 - m1) * (l1 - m1) + (l2 - m2) * (l2 - m2)) / self.var / 2

        accuracy = y * target

        loss = -F.sum(accuracy)

        r = xp.where(
            xp.argmax(y.data, axis=1) == xp.argmax(target.data, axis=1), 1, 0).reshape((num_lm, 1)).astype(xp.float32)

        loss += F.sum((r - b) * (r - b))
        bb = xp.sum(b.data) / num_lm
        lossm = self.r * (r - bb)
        loss += F.sum(Variable(lossm) * ln_p)
        return loss

    def r_space(self, x, target, num_lm, lz, step=1):
        accuracy = 0
        for i in range(step):
            lc = lz[i].reshape(num_lm, 2)

            lm = Variable(lc)
            xm = make_sampled_image.generate_xm_const_size(lm.data, self.gsize, x.data, num_lm, g_size=self.gsize,
                                                           img_size=self.img_size)
            l, y, b = self.recurrent_forward(xm, lm)
            accuracy = y.data * target.data
        return np.sum(accuracy)

    def make_img(self, x, l, num_lm, random=0):
        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
        else:
            eps = xp.random.normal(0, 1, size=l.data.shape).astype(xp.float32)
            lm = xp.clip(l.data + eps * xp.sqrt(self.vars), 0, 1)

            lm = Variable(lm.astype(xp.float32))
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_const_size_gpu(lm.data, self.gsize, x.data, num_lm, g_size=self.gsize,
                                                               img_size=self.img_size)
        else:
            xm = make_sampled_image.generate_xm_const_size(lm.data, self.gsize, x.data, num_lm, g_size=self.gsize,
                                                           img_size=self.img_size)

        return xm, lm

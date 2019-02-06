# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""
import chainer.functions as F
from chainer import Variable
import chainer.links as L
from env import xp
from modelfile.model_dram import BASE
import make_sampled_image
import math


class SAF(BASE):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, wvar=0, n_step=2, gpu_id=-1):
        super(BASE, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            # in 256 * 256 * 3
            #
            # 24 * 24 ->  12 * 12 -> 6 * 6 -> 1
            #         pool      pool      full
            cnn_1_1=L.Convolution2D(3, 32, 3, pad=1),
            cnn_1_2=L.Convolution2D(32, 32, 3, pad=1),
            cnn_2_1=L.Convolution2D(32, 64, 3, pad=1),
            cnn_2_2=L.Convolution2D(64, 64, 3, pad=1),
            cnn_3_1=L.Convolution2D(64, 64, 3, pad=1),
            cnn_3_2=L.Convolution2D(64, 64, 3, pad=1),
            full_1=L.Linear(3 * 3 * 64, 256),
            # full_2=L.Linear(None, 10),

            glimpse_loc=L.Linear(2, 256),

            norm_1_1=L.BatchNormalization(32),
            norm_1_2=L.BatchNormalization(32),
            norm_2_1=L.BatchNormalization(64),
            norm_2_2=L.BatchNormalization(64),
            norm_3_1=L.BatchNormalization(64),
            norm_3_2=L.BatchNormalization(64),
            norm_f1=L.BatchNormalization(256),

            # 記憶を用いるLSTM部分
            rnn_1=L.LSTM(n_units, n_units),
            rnn_2=L.LSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(n_units, 2),

            # 入力画像を処理するネットワーク
            # 256 * 256 -> 64 * 64 -> 32 * 32 -> 16 * 16
            #           pool
            context_cnn_1=L.Convolution2D(3, 64, 3, pad=1),
            context_cnn_2=L.Convolution2D(64, 64, 3, pad=1),
            context_cnn_3=L.Convolution2D(64, 128, 3, pad=1),
            context_cnn_4=L.Convolution2D(128, 128, 3, pad=1),
            context_cnn_5=L.Convolution2D(128, 128, 3, pad=1),
            context_full=L.Linear(16 * 16 * 128, n_units),

            l_norm_cc1=L.BatchNormalization(64),
            l_norm_cc2=L.BatchNormalization(64),
            l_norm_cc3=L.BatchNormalization(128),
            l_norm_cc4=L.BatchNormalization(128),
            l_norm_cc5=L.BatchNormalization(128),

            # baseline network 強化学習の期待値を学習し、バイアスbとする
            baseline=L.Linear(n_units, 1),

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
        self.gsize = 24
        self.train = True
        self.var = var
        if wvar == 0:
            self.vars = var
        else:
            self.vars = wvar
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = 0.5
        self.r_recognize = 1.0
        self.n_step = n_step

    def make_img(self, x, l, num_lm, random=0):
        s = xp.log10(xp.ones((1, 1)) * self.gsize / self.img_size) + 1
        sm = xp.repeat(s, num_lm, axis=0)

        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
        else:
            eps = xp.random.normal(0, 1, size=l.data.shape).astype(xp.float32)
            lm = xp.clip(l.data + eps * xp.sqrt(self.vars), 0, 1)
            lm = Variable(lm.astype(xp.float32))
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_rgb_gpu(lm.data, sm, x, num_lm, g_size=self.gsize)
        else:
            xm = make_sampled_image.generate_xm_rgb(lm.data, sm, x, num_lm, g_size=self.gsize)
        return xm, lm

    def first_forward(self, x, num_lm):
        self.rnn_1(Variable(xp.zeros((num_lm, self.n_unit)).astype(xp.float32)))
        h2 = F.relu(self.l_norm_cc1(self.context_cnn_1(F.average_pooling_2d(x, 4, stride=4))))
        h3 = F.relu(self.l_norm_cc2(self.context_cnn_2(h2)))
        h4 = F.relu(self.l_norm_cc3(self.context_cnn_3(F.max_pooling_2d(h3, 2, stride=2))))
        h5 = F.relu(self.l_norm_cc4(self.context_cnn_4(h4)))
        h6 = F.relu(self.l_norm_cc5(self.context_cnn_5(h5)))
        h7 = F.relu(self.context_full(F.max_pooling_2d(h6, 2, stride=2)))
        h8 = F.relu(self.rnn_2(h7))

        l = F.sigmoid(self.attention_loc(h8))
        b = F.sigmoid(self.baseline(Variable(h8.data)))
        return l, b

    def recurrent_forward(self, xm, lm):
        hgl = F.relu(self.glimpse_loc(lm))

        h = self.glimpse_forward(xm)
        hr1 = F.relu(self.rnn_1(hgl * h))

        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, y, b

    def glimpse_forward(self, x):
        h = F.relu(self.norm_1_1(self.cnn_1_1(x)))
        h = F.relu(self.norm_1_2(F.max_pooling_2d(self.cnn_1_2(h), 2, stride=2)))
        h = F.relu(self.norm_2_1(self.cnn_2_1(h)))
        h = F.relu(self.norm_2_2(F.max_pooling_2d(self.cnn_2_2(h), 2, stride=2)))
        h = F.relu(self.norm_3_1(self.cnn_3_1(h)))
        h = F.relu(self.norm_3_2(F.max_pooling_2d(self.cnn_3_2(h), 2, stride=2)))
        h = F.relu(self.norm_f1(self.full_1(h)))
        return h

    def set_b(self):
        self.b_log = 0

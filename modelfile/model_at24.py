# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""
import chainer.functions as F
from chainer import Variable
from env import xp
from modelfile.model_at import BASE
import math


class SAF(BASE):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, n_step=2, gpu_id=-1):
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

            glimpse_loc=L.Linear(3, 256),

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
            attention_scale=L.Linear(n_units, 1),

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
        self.gsize = 20
        self.train = True
        self.var = var
        self.vars = var
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = 0.5
        self.r_recognize = 1.0
        self.n_step = n_step

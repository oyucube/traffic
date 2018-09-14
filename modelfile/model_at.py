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
            full_1=L.Linear(4 * 4 * 64, 256),
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
        self.gsize = 32
        self.train = True
        self.var = var
        self.vars = var
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = 0.5
        self.r_recognize = 1.0
        self.n_step = n_step

    def reset(self):
        self.rnn_1.reset_state()
        self.rnn_2.reset_state()

    def set_b(self):
        self.b_log = 0

    def __call__(self, x, target, bf=False):
        self.reset()
        n_step = self.n_step
        num_lm = x.shape[0]
        if chainer.config.train:
            r_buf = 0
            l, s, b = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)

                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    r_buf += size_p
                    r = xp.where(
                        xp.argmax(y.data, axis=1) == xp.argmax(target, axis=1), 1, 0).reshape((num_lm, 1)).astype(
                        xp.float32)
                    loss *= self.r_recognize
                    loss += F.sum((r - b) * (r - b))  # loss baseline
                    k = self.r * (r - b.data)  # calculate r
                    loss += F.sum(k * r_buf)

                    return loss / num_lm
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)
                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    r_buf += size_p
                l = l1
                s = s1
                b = b1

        else:
            l, s, b1 = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    if bf:
                        self.b_log += xp.sum(b1.data)
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
                    accuracy = xp.sum(y.data * target)
                    if self.use_gpu:
                        accuracy = chainer.cuda.to_cpu(accuracy)
                    return accuracy
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
                l = l1
                s = s1

    def use_model(self, x, t):
        self.reset()
        num_lm = x.shape[0]
        n_step = self.n_step
        s_list = xp.empty((n_step, num_lm, 1))
        l_list = xp.empty((n_step, num_lm, 2))
        l, s, b1 = self.first_forward(x, num_lm)
        for i in range(n_step):
            if i + 1 == n_step:
                xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
                s_list[i] = sm.data
                l_list[i] = lm.data
                accuracy = y.data * t
                s_list = xp.power(10, s_list - 1)
                return xp.sum(accuracy, axis=1), l_list, s_list
            else:
                xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
            l = l1
            s = s1
            s_list[i] = s.data
            l_list[i] = l.data
        return

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
        s = F.sigmoid(self.attention_scale(h8))
        b = F.sigmoid(self.baseline(Variable(h8.data)))
        return l, s, b

    def recurrent_forward(self, xm, lm, sm):
        ls = xp.concatenate([lm.data, sm.data], axis=1)
        hgl = F.relu(self.glimpse_loc(Variable(ls)))

        h = self.glimpse_forward(xm)
        hr1 = F.relu(self.rnn_1(hgl * h))

        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        s = F.sigmoid(self.attention_scale(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, s, y, b

    def glimpse_forward(self, x):
        h = F.relu(self.norm_1_1(self.cnn_1_1(x)))
        h = F.relu(self.norm_1_2(F.max_pooling_2d(self.cnn_1_2(h), 2, stride=2)))
        h = F.relu(self.norm_2_1(self.cnn_2_1(h)))
        h = F.relu(self.norm_2_2(F.max_pooling_2d(self.cnn_2_2(h), 2, stride=2)))
        h = F.relu(self.norm_3_1(self.cnn_3_1(h)))
        h = F.relu(self.norm_3_2(F.max_pooling_2d(self.cnn_3_2(h), 2, stride=2)))
        h = F.relu(self.norm_f1(self.full_1(h)))
        return h

    # loss 関数を計算
    def cul_loss(self, y, target, l, s, lm, sm):

        zm = xp.power(10, sm.data - 1)

        l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
        m1, m2 = F.split_axis(lm, indices_or_sections=2, axis=1)
        ln_p = ((l1 - m1) * (l1 - m1) + (l2 - m2) * (l2 - m2)) / self.var / zm / zm / 2
        # size
        size_p = (sm - s) * (sm - s) / self.vars + ln_p

        accuracy = y * target

        loss = -F.sum(accuracy)
        return loss, size_p

    def make_img(self, x, l, s, num_lm, random=0):
        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
            sm = Variable(xp.clip(s.data, 0, 1))
        else:
            eps = xp.random.normal(0, 1, size=l.data.shape).astype(xp.float32)
            epss = xp.random.normal(0, 1, size=s.data.shape).astype(xp.float32)
            sm = xp.clip((s.data + xp.sqrt(self.var) * epss), 0, 1).astype(xp.float32)
            lm = xp.clip(l.data + xp.power(10, sm - 1) * eps * xp.sqrt(self.vars), 0, 1)
            sm = Variable(sm)
            lm = Variable(lm.astype(xp.float32))
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_rgb_gpu(lm.data, sm.data, x, num_lm, g_size=self.gsize)
        else:
            xm = make_sampled_image.generate_xm_rgb(lm.data, sm.data, x, num_lm, g_size=self.gsize)
        return xm, lm, sm

    def s1_determin(self, x, t, l1, s1):
        self.reset()
        num_lm = x.data.shape[0]
        s1 = xp.power(10, s1 - 1)
        self.first_forward(x, num_lm)
        xm, lm, sm = self.make_img(x, Variable(l1), Variable(s1), num_lm, random=0)
        l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
        accuracy = y.data * t.data
        return xp.sum(accuracy)

    # def s2_determin(self, x, t, l2, s2):
    #     self.reset()
    #     n_step = self.n_step
    #     num_lm = x.data.shape[0]
    #
    #     l, s, b1 = self.first_forward(x, num_lm)
    #     xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
    #     l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
    #     # 画像作成の場所を指定
    #     s2 = xp.power(10, s2 - 1)
    #     xm, lm, sm = self.make_img(x, Variable(l2), Variable(s2), num_lm, random=0)
    #     l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
    #     accuracy = y.data * t.data
    #     return xp.sum(accuracy)

class SAF(BASE):
    pass

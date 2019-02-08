# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""
import chainer.functions as F
from chainer import Variable
import chainer.links as L
from env import xp
from modelfile.model_at import BASE
import make_sampled_image
import chainer


class BASEM(BASE):
    def __init__(self, n_units=128, n_out=0, img_size=112, var=0.18, wvar=0, n_step=2, gpu_id=-1):
        super(BASE, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            glimpse_cnn_1=L.Convolution2D(1, 20, 5),  # in 20 out 16
            glimpse_cnn_2=L.Convolution2D(20, 40, 5),  # in 16 out 12
            glimpse_cnn_3=L.Convolution2D(40, 80, 5),  # in 12 out 8
            glimpse_full=L.Linear(8 * 8 * 80, n_units),
            glimpse_loc=L.Linear(2, n_units),

            # baseline network 強化学習の期待値を学習し、バイアスbとする
            baseline=L.Linear(n_units, 1),

            l_norm_c1=L.BatchNormalization(20),
            l_norm_c2=L.BatchNormalization(40),
            l_norm_c3=L.BatchNormalization(80),

            # 記憶を用いるLSTM部分
            rnn_1=L.LSTM(n_units, n_units),
            rnn_2=L.LSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(n_units, 2),
            attention_scale=L.Linear(n_units, 1),

            # 入力画像を処理するネットワーク
            context_cnn_1=L.Convolution2D(1, 2, 5),  # 56 to 52 pooling: 26
            context_cnn_2=L.Convolution2D(2, 2, 5),  # 26 to 22 pooling
            context_cnn_3=L.Convolution2D(2, 2, 4),  # 22 to 16

            l_norm_cc1=L.BatchNormalization(2),
            l_norm_cc2=L.BatchNormalization(2),
            l_norm_cc3=L.BatchNormalization(2),

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

    def first_forward(self, x, num_lm, test=False):
        self.rnn_1(Variable(xp.zeros((num_lm, self.n_unit)).astype(xp.float32)))
        h2 = F.relu(self.l_norm_cc1(self.context_cnn_1(F.max_pooling_2d(x, 2, stride=2))))
        h3 = F.relu(self.l_norm_cc2(self.context_cnn_2(F.max_pooling_2d(h2, 2, stride=2))))
        h4 = F.relu(self.l_norm_cc3(self.context_cnn_3(F.max_pooling_2d(h3, 2, stride=2))))
        h5 = F.relu(self.rnn_2(h4))

        l = F.sigmoid(self.attention_loc(h5))
        s = F.sigmoid(self.attention_scale(h5))
        b = F.sigmoid(self.baseline(Variable(h5.data)))
        return l, s, b

    def recurrent_forward(self, xm, lm, sm, test=False):
        hgl = F.relu(self.glimpse_loc(lm))
        hg1 = F.relu(self.l_norm_c1(self.glimpse_cnn_1(Variable(xm))))
        hg2 = F.relu(self.l_norm_c2(self.glimpse_cnn_2(hg1)))
        hg3 = F.relu(self.l_norm_c3(self.glimpse_cnn_3(hg2)))
        hgf = F.relu(self.glimpse_full(hg3))

        hr1 = F.relu(self.rnn_1(hgl * hgf))
        # ベクトルの積
        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        s = F.sigmoid(self.attention_scale(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, s, y, b

    def make_img(self, x, l, s, num_lm, random=0):
        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
            sm = Variable(xp.clip(s.data, 0, 1))
        else:
            eps = xp.random.normal(0, 1, size=l.data.shape).astype(xp.float32)
            epss = xp.random.normal(0, 1, size=s.data.shape).astype(xp.float32)
            sm = xp.clip((s.data + xp.sqrt(self.vars) * epss), 0, 1).astype(xp.float32)
            lm = xp.clip(l.data + xp.power(10, sm - 1) * eps * 2 * xp.sqrt(self.vars), 0, 1)
            sm = Variable(sm)
            lm = Variable(lm.astype(xp.float32))
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_gpu(lm.data, sm.data, x, num_lm, g_size=self.gsize)
        else:
            xm = make_sampled_image.generate_xm(lm.data, sm.data, x, num_lm, g_size=self.gsize)
        return xm, lm, sm

    def __call__(self, x, target, bf=False):
        self.reset()
        n_step = self.n_step
        num_lm = x.shape[0]
        loss_func = 0
        if chainer.config.train:
            l, s, b = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)

                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    loss_func += self.r_recognize * loss
                    r = xp.where(
                        xp.argmax(y.data, axis=1) == xp.argmax(target, axis=1), 1, 0).reshape((num_lm, 1)).astype(
                        xp.float32)
                    loss_func += F.sum((r - b) * (r - b))  # loss baseline
                    k = self.r * (r - b.data)  # calculate r
                    loss_func += F.sum(k * size_p)

                    return loss_func / num_lm
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)
                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    loss_func += self.r_recognize * loss
                    r = xp.where(
                        xp.argmax(y.data, axis=1) == xp.argmax(target, axis=1), 1, 0).reshape((num_lm, 1)).astype(
                        xp.float32)
                    loss_func += F.sum((r - b) * (r - b))  # loss baseline
                    k = self.r * (r - b.data)  # calculate r
                    loss_func += F.sum(k * size_p)
                l = l1
                s = s1
                b = b1


class SAF(BASEM):
    pass

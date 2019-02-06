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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import chainer
from chainer import cuda, serializers
import sys
from tqdm import tqdm
import datetime
import importlib
import socket
from PIL import Image, ImageDraw, ImageFont


def get_batch(ds, index, repeat):
    nt = ds.num_target
    # print(index)
    batch_size = index.shape[0]
    return_x = np.empty((batch_size, 3, 256, 256))
    return_t = np.zeros((batch_size, nt))
    for bi in range(batch_size):
        return_x[bi] = ds[index[bi]][0]
        return_t[bi] = ds[index[bi]][1]
    return_x = return_x.reshape(batch_size, 3, 256, 256).astype(np.float32)
    return_t = return_t.astype(np.float32)
    return_x = xp.asarray(xp.tile(return_x, (repeat, 1, 1, 1)))
    return_t = xp.asarray(xp.tile(return_t, (repeat, 1)))
    return return_x, return_t


def draw_attention(d_img, d_l_list, d_s_list, index, save="", acc=""):
    draw = ImageDraw.Draw(d_img)
    color_list = ["red", "yellow", "blue", "green"]
    size = 256
    for j in range(l_list.shape[0]):
        l = d_l_list[j][index]
        s = d_s_list[j][index]
        print(l)
        p1 = (size * (l - s / 2))
        p2 = (size * (l + s / 2))
        # p1[0] = size - p1[0]
        # p2[0] = size - p2[0]
        print([p1[0], p1[1], p2[0], p2[1]])
#        draw.rectangle([p1[0], p1[1], p2[0], p2[1]], outline=color_list[j])
        draw.rectangle([p1[1], p1[0], p2[1], p2[0]], outline=color_list[j])

    if len(acc) > 0:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\msgothic.ttc", 20)
        draw.text([120, 230], acc, font=font, fill="red")
    if len(save) > 0:
        img.save(save + ".png")
    return d_img

#  引数分解
parser = argparse.ArgumentParser()
# load model id

# * *********************************************    config    ***************************************************** * #
parser.add_argument("-a", "--am", type=str, default="model_at24",
                    help="attention model")
parser.add_argument("-l", "--l", type=str, default="best_p24_06try2_B",
                    help="load model name")
parser.add_argument("-d", "--data", type=str, default="m5class",
                    help="data")
test_b = 1
num_step = 1

# * **************************************************************************************************************** * #

parser.add_argument("-b", "--batch_size", type=int, default=50,
                    help="batch size")
parser.add_argument("-e", "--epoch", type=int, default=50,
                    help="iterate training given epoch times")
parser.add_argument("-m", "--num_l", type=int, default=40,
                    help="a number of sample ")
parser.add_argument("-s", "--step", type=int, default=2,
                    help="look step")
parser.add_argument("-v", "--var", type=float, default=0.02,
                    help="sample variation")
parser.add_argument("-g", "--gpu", type=int, default=-1,
                    help="use gpu")
# train id
parser.add_argument("-i", "--id", type=str, default="sample",
                    help="data id")

# model save id
parser.add_argument("-o", "--filename", type=str, default="v1",
                    help="prefix of output file names")
args = parser.parse_args()

file_id = args.filename
model_id = args.id
num_lm = args.num_l
n_epoch = args.epoch
train_id = args.id

train_b = args.batch_size
train_var = args.var
gpu_id = args.gpu
model_file_name = args.am

# naruto ならGPUモード
if socket.gethostname() == "chainer":
    gpu_id = 0
    log_dir = "/home/y-murata/storage/traffic/"
    data_dir = "/home/y-murata/traffic/data/"
else:
    data_dir = "C:/Users/waka-lab/Documents/data/data/"
    log_dir = "log/"
# load data
if args.data == "5class" or args.data == "m5class":
    data_dir = data_dir + "newdata/"
else:
    data_dir = data_dir + "origin/"
dl = importlib.import_module("dataset." + args.data)
train_data = dl.MyDataset(data_dir, "train")
val_data = dl.MyDataset(data_dir, "test")

xp = cuda.cupy if gpu_id >= 0 else np

data_max = len(train_data)
test_max = len(val_data)
num_val = 1000
num_val_loop = 10  # val loop 10 times
img_size = 256
n_target = train_data.num_target
num_class = n_target
target_c = ""

# モデルの作成
mf = importlib.import_module("modelfile." + model_file_name)
model = mf.SAF(n_out=n_target, img_size=img_size, var=train_var, n_step=num_step, gpu_id=gpu_id)
# model load
if len(args.l) != 0:
    serializers.load_npz('model/' + args.l + '.model', model)


# gpuの設定
if gpu_id >= 0:
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()

pp = xp.array(range(len(val_data)))
pp = xp.random.permutation(pp)

# nd = xp.array(range(100))
nd = pp
sample = test_b
det = 10
a_size = 0.3
space1 = xp.zeros((sample, det, det))
space2 = xp.zeros((sample, det, det))
for s in tqdm(range(sample)):
    for i in range(det):
        for j in range(det):
            l2 = xp.array([[i / det, j / det]]).astype("float32")
            s2 = xp.array([[a_size]]).astype("float32")
            with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
                x, t = get_batch(val_data, xp.array([nd[s]]), 1)
                space1[s][i][j] = model.s1_determin(x, t, l2, s2)
a_size = 0.5
for s in tqdm(range(sample)):
    for i in range(det):
        for j in range(det):
            l2 = xp.array([[i / det, j / det]]).astype("float32")
            s2 = xp.array([[a_size]]).astype("float32")
            with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
                x, t = get_batch(val_data, xp.array([nd[s]]), 1)
                space2[s][i][j] = model.s1_determin(x, t, l2, s2)



# 描画
with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
    x, t = get_batch(val_data, nd[0:sample], 1)
    acc, l_list, s_list, x_list = model.use_model(x, t)
print(acc)

for i in range(sample):
    plt.figure(dpi=200)
#     print(nd[i])
    img = val_data.get_image(nd[i])
    acc_str = ("{:1.8f}".format(acc[i]))
    print(acc_str)
    print(acc[i])
    draw = draw_attention(img, l_list, s_list, i, save="", acc=acc_str[0:6])
    plt.subplot(221)
    plt.imshow(np.asarray(draw))
    plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
    plt.tick_params(labelleft="off", left="off")  # y軸の削除

    plt.subplot(223)
    plt.title("size=3")
    plt.xlabel("position X")
    plt.ylabel("position Y")
    plt.pcolor(np.flipud(space1[i]), cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.plot([1.5, 8.5], [1.5, 1.5], 'r--', lw=1)
    plt.plot([1.5, 8.5], [8.5, 8.5], 'r--', lw=1)
    plt.plot([1.5, 1.5], [1.5, 8.5], 'r--', lw=1)
    plt.plot([8.5, 8.5], [1.5, 8.5], 'r--', lw=1)
    plt.colorbar()

    plt.subplot(224)
    plt.title("size=5")
    plt.xlabel("position X")
    plt.ylabel("position Y")
    plt.pcolor(np.flipud(space2[i]), cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar()
    plt.plot([2.5, 7.5], [2.5, 2.5], 'r--', lw=1)
    plt.plot([2.5, 7.5], [7.5, 7.5], 'r--', lw=1)
    plt.plot([2.5, 2.5], [2.5, 7.5], 'r--', lw=1)
    plt.plot([7.5, 7.5], [2.5, 7.5], 'r--', lw=1)
    plt.savefig("buf/space" + str(i) + ".png")
    plt.close()

#     image                graph
#        x
#     -------→           ↑
#　　 l                   l
#　y  l                 y l
#　　 l                   l
#     ⇓                   -------→
#                              x
#      y軸を反転
#
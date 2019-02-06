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
import argparse
import chainer
from chainer import cuda, serializers
import matplotlib.pyplot as plt
import seaborn as sns;
import importlib
import socket
from mylib.my_functions import get_batch, draw_attention, get_batch_mnist, draw_attention_xm
import matplotlib
import importlib
from tqdm import tqdm
import os
matplotlib.use('Agg')


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
#  引数分解


parser = argparse.ArgumentParser()
# load model id

# * *********************************************    config    ***************************************************** * #
parser.add_argument("-a", "--am", type=str, default="model_at24",
                    help="attention model")
# data selection
parser.add_argument("-d", "--data", type=str, default="m5class",
                    help="data")
parser.add_argument("-l", "--l", type=str, default="best_normal_try1",
                    help="load model name")
test_b = 250
num_step = 1
out_dir_name = "5classreward"
# * **************************************************************************************************************** * #

# hyper parameters
parser.add_argument("-e", "--epoch", type=int, default=30,
                    help="iterate training given epoch times")
parser.add_argument("-b", "--batch_size", type=int, default=20,
                    help="batch size")
parser.add_argument("-m", "--num_l", type=int, default=30,
                    help="a number of sample ")
parser.add_argument("-s", "--step", type=int, default=2,
                    help="look step")
parser.add_argument("-v", "--var", type=float, default=0.02,
                    help="sample variation")
parser.add_argument("-g", "--gpu", type=int, default=-1,
                    help="use gpu")
# log config
parser.add_argument("-o", "--filename", type=str, default="",
                    help="prefix of output file names")
parser.add_argument("-p", "--logmode", type=int, default=1,
                    help="log mode")
args = parser.parse_args()

file_id = args.filename
num_lm = args.num_l
n_epoch = args.epoch
train_b = args.batch_size
train_var = args.var
gpu_id = args.gpu
crop = 1

# naruto ならGPUモード
if socket.gethostname() == "chainer":
    gpu_id = 0
    log_dir = "/home/y-murata/storage/traffic/"
    data_dir = "/home/y-murata/traffic/data/"
else:
    data_dir = "C:/Users/waka-lab/Documents/data/data/"
    log_dir = "log/"
# load data
# load data
data_dir = data_dir + "reward/"

dl = importlib.import_module("dataset." + args.data)
# train_data = dl.MyDataset(data_dir, "train")
# val_data = dl.MyDataset(data_dir, "test")
val_data = dl.MyDataset(data_dir, "label")


xp = cuda.cupy if gpu_id >= 0 else np

# data_max = train_data.len
test_max = len(val_data)
num_val = 100
num_val_loop = 10  # val loop 10 times
#
# data_max = 1000
# test_max = 1000
# num_val = 100
# num_val_loop = 1  # val loop 10 times


img_size = 256
n_target = val_data.num_target
num_class = n_target
target_c = ""
# test_b = test_max

# モデルの作成
model_file_name = args.am

sss = importlib.import_module("modelfile." + model_file_name)
model = sss.SAF(n_out=n_target, img_size=img_size, var=train_var, n_step=num_step, gpu_id=gpu_id)

# model load
if len(args.l) != 0:
    print("load model model/best{}.model".format(args.l))
    serializers.load_npz('model/' + args.l + '.model', model)
else:
    print("must load model!!!")
    exit()

# オプティマイザの設定
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# gpuの設定
if gpu_id >= 0:
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()


pp = xp.array(range(len(val_data)))

nd = pp
sample = test_b
det = 20
a_size = 0.3
space1 = xp.zeros((sample, det, det))
for s in tqdm(range(sample)):
    for i in range(det):
        for j in range(det):
            l2 = xp.array([[i / det, 0.5]]).astype("float32")
            s2 = xp.array([[j / det]]).astype("float32")
            with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
                x, t = get_batch(val_data, xp.array([nd[s]]), 1)
                space1[s][i][j] = model.s1_determin(x, t, l2, s2)

reward_a = np.flipud(space1.sum(axis=0) / test_b)
plt.figure()
plt.pcolor(reward_a, cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.savefig("buf/reward_m5class.png")
plt.close()

np.save("buf/reward_m5class", np.flipud(space1.sum(axis=0) / test_b))
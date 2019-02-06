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
import importlib
import socket
from mylib.my_functions import get_batch, draw_attention, get_batch_mnist, draw_attention_xm
import matplotlib
import importlib
import os
matplotlib.use('Agg')


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
test_b = 100
num_step = 1
out_dir_name = "no_acc"
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
if args.data == "5class" or args.data == "m5class":
    data_dir = data_dir + "newdata/"
elif args.data == "art":
    data_dir = data_dir + "origin/"
elif args.data == "art3":
    data_dir = data_dir + "origin4/"
elif args.data == "difd":
    data_dir = data_dir + "difdata/"
elif args.data == "multsign":
    data_dir = data_dir + "multsign/"
elif args.data == "easymult":
    data_dir = data_dir + "easymult/"
else:
    print("unknown dataset")
    exit()
dl = importlib.import_module("dataset." + args.data)
# train_data = dl.MyDataset(data_dir, "train")
# val_data = dl.MyDataset(data_dir, "test")
train_data = dl.MyDataset(data_dir, "train")
val_data = dl.MyDataset(data_dir, "train")


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
    print("load model!!!")
    exit()
# model load
# if len(args.l) != 0:
#     print("load model model/my{}{}.model".format(args.l, model_id))
#     serializers.load_npz('model/my' + args.l + mod el_id + '.model', model)

# オプティマイザの設定
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# gpuの設定
if gpu_id >= 0:
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()


perm = np.random.permutation(test_max)
with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
    x, t = get_batch(val_data, perm[0:test_b], 1)
#     x, t = get_batch_mnist(val_data, perm[0:test_b], 1, image_size=256)

    acc, l_list, s_list, x_list = model.use_model(x, t)

# print("acc")
# print(acc)
# print("l_list")
# print(l_list)
print("s_list")
# print(s_list)
os.makedirs("buf/" + out_dir_name)
save_dir = "buf/" + out_dir_name + "/attention"
for i in range(test_b):
    save_filename = save_dir + str(i)
    # print(acc[i])
    img = val_data.get_image(perm[i])
    acc_str = ("{:1.8f}".format(acc[i]))
    acc_str = ""
    # print(acc_str)
    # print(l_list[0][i])
    # print(l_list[1][i])
    # print(s_list[0][i])
    # draw_attention(img, l_list, s_list, i, save=save_filename, acc=acc_str)
    draw_attention_xm(img, l_list, s_list, x_list, i, save=save_filename, acc=acc_str)

#    draw_attention(img, l_list, s_list, i, save=save_filename, acc=acc_str[0:6])
print("average of acc")
print(np.mean(acc))


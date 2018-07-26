# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:46:33 2016

@author: oyu
"""
import os
# os.environ["CHAINER_TYPE_CHECK"] = "0" #ここでオフに  オンにしたかったら1にするかコメントアウト
import numpy as np
# i = np.random()
# np.random.seed()
import argparse
import chainer
from chainer import cuda, serializers
from tqdm import tqdm
import datetime
import importlib
import socket
import gc
from mylib.my_functions import copy_model
from mylib.my_logger import LOGGER

parser = argparse.ArgumentParser()

# model selection
parser.add_argument("-a", "--am", type=str, default="model_dram",
                    help="attention model")
# data selection
parser.add_argument("-d", "--data", type=str, default="mnist",
                    help="data")
# hyper parameters
parser.add_argument("-e", "--epoch", type=int, default=50,
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
# load model id
parser.add_argument("-l", "--l", type=str, default="",
                    help="load model name")
# log config
parser.add_argument("-o", "--filename", type=str, default="",
                    help="prefix of output file names")
parser.add_argument("-q", "--logmode", type=int, default=1,
                    help="log mode")
parser.add_argument("-p", "--pre", type=str, default="",
                    help="pre train")

args = parser.parse_args()

model_file_name = args.am
file_id = args.filename
num_lm = args.num_l
n_epoch = args.epoch
num_step = args.step
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
data_dir = data_dir + "mnistdata/"
dl = importlib.import_module("dataset." + args.data)
train_data = dl.MyDataset(data_dir)
val_data = dl.MyDataset(data_dir)

xp = cuda.cupy if gpu_id >= 0 else np

data_max = len(train_data)
test_max = len(val_data)
num_val = 1000
num_val_loop = 10  # val split
img_size = 112
n_target = train_data.num_target
num_class = n_target
target_c = ""

# モデルの作成
mf = importlib.import_module("modelfile." + model_file_name)
model = mf.SAF(n_out=n_target, img_size=img_size, var=train_var, n_step=num_step, gpu_id=gpu_id)


# log setting
if file_id == "":
    file_id = datetime.datetime.now().strftime("%m%d%H%M%S")
log_dir = log_dir + file_id + "/"
os.mkdir(log_dir)
logger = LOGGER(log_dir, file_id, n_epoch=n_epoch)

logger.l_print("{} class recognition\nclass:{} use {}".format(num_class, target_c, dl.data_name))
logger.l_print("model:{}".format(model_file_name))
logger.l_print("parameter\n")
logger.l_print("step:{} sample:{} batch_size:{} var:{} crop:{}".format(num_step, num_lm, train_b, train_var, crop))
logger.l_print("log dir:{}".format(log_dir))
logger.l_print("going to train {} epoch".format(n_epoch))

# model load
if len(args.l) != 0:
    logger.l_print("load model model/{}.model".format(args.l))
    serializers.load_npz('model/' + args.l + '.model', model)
logger.update_log()

# オプティマイザの設定
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# gpuの設定
if gpu_id >= 0:
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()

val_batch_size = int(num_val / num_val_loop)
train_iterator = chainer.iterators.SerialIterator(train_data, train_b, shuffle=True)
e_val_iterator = chainer.iterators.SerialIterator(val_data, val_batch_size, shuffle=True)
e_train_iterator = chainer.iterators.SerialIterator(train_data, val_batch_size, shuffle=True)

# train
for epoch in range(n_epoch):
    print("(epoch: {})\n".format(epoch + 1))
    for i in tqdm(range(0, data_max, train_b), ncols=60):
        model.cleargrads()
        with chainer.using_config('train', True):
            x, t = chainer.dataset.concat_examples(train_iterator.next(), device=gpu_id)
            x = xp.tile(x, (num_lm, 1, 1, 1))
            t = xp.tile(t, (num_lm, 1))
            loss = model(x, t)
            logger.set_loss(loss.data, epoch)
            loss.backward()
            loss.unchain_backward()  # truncate
        optimizer.update()
        del x, t, loss
        gc.collect()

    # evaluate
    train_acc = 0
    val_acc = 0
    for i in range(0, num_val, val_batch_size):
        with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
            x, t = chainer.dataset.concat_examples(e_train_iterator.next(), device=gpu_id)
            train_acc += model(x, t)
            x, t = chainer.dataset.concat_examples(e_val_iterator.next(), device=gpu_id)
            val_acc += model(x, t)

    # save accuracy
    logger.set_acc(train_acc / num_val, val_acc / num_val, epoch)
    logger.save_acc()
    logger.update_log()
    # save model
    if gpu_id >= 0:
        serializers.save_npz(log_dir + "/" + logger.best + file_id + '.model', model.to_cpu())
        model.to_gpu()
    else:
        serializers.save_npz(log_dir + "/" + logger.best + file_id + '.model', model)

# logger.l_print("last acc:{}  max_acc:{}\n".format(acc1_array[n_epoch - 1], max_acc))

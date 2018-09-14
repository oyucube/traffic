import cv2
import anno_func
import json
import math
import random
import numpy as np
from PIL import Image
import os


# add_data(out file name,source image, target_sign)
def add_data(img_o, source, sign):
    s_p = np.random.randint(0, 400 - 256, size=(2, ))
    img = source.resize((400, 400))
    img = img.crop((s_p[0], s_p[1], s_p[0] + 256, s_p[1] + 256))

    size = int(min_size * math.exp(math.log(max_size / min_size) * random.random()))
    expand = 0.7 + 0.6 * random.random()
#     rotate = random.randint(0, 60)
    a_p = np.random.randint(0, 256 - size, size=(2, ))
    sign = sign.resize((size, int(size * expand)))
    mask = sign.split()[3]
    img.paste(sign, list(a_p), mask)
    img.save(datadir + "origin/" + img_o)
    return 0


# http://cg.cs.tsinghua.edu.cn/traffic-sign/
# %matplotlib inline
# source data dir
datadir = "C:/Users/waka-lab/Documents/data/data/"
base = os.path.dirname(os.path.abspath(__file__))
jpgdir = os.path.normpath(os.path.join(base, './dataset/sign/'))
# jpgdir = "C:/Users/waka-lab\Google ドライブ/research/traffic/dataset/sign"
json_fp = open('annotations.json')
jf = json.loads(json_fp.read())

# output dir
savedir = datadir + "origin/"
f_log = open(savedir + 'log.txt', 'w')
f_label = open(savedir + 'label.txt', 'w')
f_train = open(savedir + 'train.txt', 'w')
f_test = open(savedir + 'test.txt', 'w')

# config
min_size = 48
max_size = 128
test_out = 100
train_out = 2000

blue_list = ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10", "i11", "i12", "i13", "i14", "i15", "50", "ip"]
target_counter = {"i9": 0, "i10": 0, "i11": 0, "i12": 0, "i13": 0, "i14": 0}
target_test = {"i9": 0, "i10": 0, "i11": 0, "i12": 0, "i13": 0, "i14": 0}
train_id = 0
test_id = 0

target_id = 0
target_list = ["i9", "i10", "i11", "i12", "i13", "i14"]
# if debug is true , stop making data at 100
debug = False
loss_data_count = 0
cropped_image_count = 0
train_list = []
train = True
end_f = False
for i in range(10):
    for img_id in jf['imgs']:
        img_an = jf['imgs'].get(img_id)
        path = img_an['path']
        img_pl_counter = 0
        obj_p = ""
    # if there are no blue traffic sign ,make data
        use_f = True
        for img_obj in img_an['objects']:
            obj = img_obj["category"]
            if obj in blue_list:
                use_f = False
                break
    # test
        if use_f and (not train):
            s_img = Image.open(datadir + path)
            a_img = Image.open(jpgdir + "\\" + target_list[target_id] + ".png")
            add_data("test" + str(test_id) + ".jpg", s_img, a_img)
            test_id += 1
            target_test[target_list[target_id]] += 1
            if target_test[target_list[target_id]] > test_out:
                target_id += 1
                if target_id == 6:
                    end_f = True
                    break
    # train
        if use_f and train:
            s_img = Image.open(datadir + path)
            a_img = Image.open(jpgdir + "\\" + target_list[target_id] + ".png")
            add_data(str(train_id) + ".jpg", s_img, a_img)
            train_id += 1
            target_counter[target_list[target_id]] += 1
            if target_counter[target_list[target_id]] > train_out:
                target_id += 1
                if target_id == 6:
                    target_id = 0
                    train = False
    if end_f:
        break

f_train.close()
f_test.close()



import cv2
import anno_func
import json
import math
import random
import numpy as np
from PIL import Image, ImageEnhance
import os


def add_data(img_o, out_id, obj_pp, label):
    size = obj_pp["bbox"]["ymax"] - obj_pp["bbox"]["ymin"]
    sr = int(size * (1 + 0.5 * random.random()))

    dif_s = sr - int(size)
    if size < obj_pp["bbox"]["xmax"] - obj_pp["bbox"]["xmin"]:
        size = obj_pp["bbox"]["xmax"] - obj_pp["bbox"]["xmin"]
    place_crop = np.array([obj_pp["bbox"]["xmin"], obj_pp["bbox"]["ymin"]])
    size_after = 24

    dx = random.randint(0, dif_s)
    dy = random.randint(0, dif_s)
    # check crop area
    img_o = img_o.crop((int(place_crop[0]) - dx, int(place_crop[1]) - dy, int(place_crop[0] + sr - dx)
                        , int(place_crop[1]) + sr - dy))
    img_o = img_o.resize((24, 24), Image.LANCZOS)
    img_o.save(savedir + out_id + ".jpg")
    f_log.write(out_id + ".jpg, " + obj_pp["category"] + "\n")
    f_log.write(str(obj_pp) + "\n")
    f_log.write("place_base" + str(place_crop) + "\n")
    f_log.write("size_after" + str(size_after) + "\n")
    f_log.write("size_crop" + str(size) + "\n")
    f_log.write("place_crop" + str(place_crop) + "\n")
    # label file format
    # image_path, label
    f_label.write(out_id + ".jpg, " + obj_pp["category"] + "\n")
    if str(out_id).startswith("test"):
        f_test.write(str(out_id) + ".jpg, " + str(label) + "\n")
    else:
        f_train.write(str(out_id) + ".jpg, " + str(label) + "\n")
    return 1


def add_artificial(img_o, out_id, obj_pp, img_sign, label):
    size = np.random.randint(64, 192, 1)
    position = (1 - size / 256) * np.random.rand(2)

    img_o = img_o.resize((256, 256), Image.LANCZOS)

    expand = 0.7 + 0.6 * random.random()
    a_p = (int(position[0] * 256), int(position[1] * 256))
    sign = img_sign.resize((size[0], int(size[0] * expand)))
    mask = sign.split()[3]
    brightness = 0.3 + 0.5 * random.random()
    sign = ImageEnhance.Brightness(sign).enhance(brightness)
    img_o.paste(sign, a_p, mask)

    sr = int(size * (1 + 0.5 * random.random()))

    dif_s = sr - int(size)
    place_crop = np.array((a_p[0], a_p[1]))
    size_after = 24

    dx = random.randint(0, dif_s)
    dy = random.randint(0, dif_s)
    # check crop area
    img_o = img_o.crop((int(place_crop[0]) - dx, int(place_crop[1]) - dy, int(place_crop[0] + sr - dx)
                        , int(place_crop[1]) + sr - dy))
    img_o = img_o.resize((24, 24), Image.LANCZOS)

    img_o.save(savedir + out_id + ".jpg")
    # label file format
    # image_path, label
    f_label.write(out_id + ".jpg, " + obj_pp["category"] + "\n")
    if str(out_id).startswith("test"):
        f_test.write(str(out_id) + ".jpg, " + str(label) + "\n")
    else:
        f_train.write(str(out_id) + ".jpg, " + str(label) + "\n")
    return 1


# source data dir
datadir = "C:/Users/waka-lab/Documents/data/data/"
json_fp = open('annotations.json')
jf = json.loads(json_fp.read())
base = os.path.dirname(os.path.abspath(__file__))
jpgdir = os.path.normpath(os.path.join(base, './dataset/sign/'))
sign_list = ["30.png", "40.png", "50.png", "60.png", "80.png"]
# output dir
savedir = datadir + "pretrain_sp2/"
f_log = open(savedir + 'log.txt', 'w')
f_label = open(savedir + 'label.txt', 'w')
f_test = open(savedir + 'test.txt', 'w')
f_train = open(savedir + 'train.txt', 'w')
# config
min_size = 64
max_size = 128
max_data = 3000

sign_t = {"pl30": "30.png", "pl40": "40.png", "pl50": "50.png", "pl60": "60.png", "pl80": "80.png"}
counter_v2 = {"pl30": 0, "pl40": 0, "pl50": 0, "pl60": 0, "pl80": 0, "pl100": 0}
counter = {"sum": 0}
for key in jf['types']:
    counter[key] = 0
# if debug is true , stop making data at 100
debug = True
loss_data_count = 0
cropped_image_count = 0

train_list = []

for img_id in jf['imgs']:
    img_an = jf['imgs'].get(img_id)
    path = img_an['path']
    img_pl_counter = 0
    obj_p = ""
    # if there are only one pl traffic sign ,make data
    for img_obj in img_an['objects']:
        obj = img_obj["category"]
        if obj == "pl30" or obj == "pl40" or obj == "pl50" or obj == "pl60" or obj == "pl80":
            img_pl_counter += 1
            obj_p = img_obj

    if img_pl_counter == 1 and counter_v2[obj_p["category"]] < 50:
        img = Image.open(datadir + path)
        f = random.randint(1, 10)
        if f == 1:
            im_sign = Image.open(jpgdir + "/" + sign_t[obj_p["category"]])
            if add_artificial(img, "test" + str(cropped_image_count), obj_p, im_sign, obj_p["category"]) == 1:
                cropped_image_count += 1
                counter_v2[obj_p["category"]] = counter_v2[obj_p["category"]] + 1
        else:
            if add_data(img, "test" + str(cropped_image_count), obj_p, obj_p["category"]) == 1:
                cropped_image_count += 1
                counter_v2[obj_p["category"]] = counter_v2[obj_p["category"]] + 1
    else:
        train_list.append(img_id)

for inf in range(8):
    for img_id in train_list:
        img_an = jf['imgs'].get(img_id)
        path = img_an['path']
        img_pl_counter = 0
        obj_p = ""
    # if there are only one pl traffic sign, make data
        for img_obj in img_an['objects']:
            obj = img_obj["category"]
            if obj == "pl30" or obj == "pl40" or obj == "pl50" or obj == "pl60" or obj == "pl80":
                img_pl_counter += 1
                obj_p = img_obj

        if img_pl_counter == 1 and counter_v2[obj_p["category"]] < max_data:
            if not(obj_p["category"] == "pl30" or obj_p["category"] == "pl40" or obj_p["category"] == "pl50"
                    or obj_p["category"] == "pl60" or obj_p["category"] == "pl80" or obj_p["category"] == "pl100"):
                print("error")
                print(path)
            img = Image.open(datadir + path)
            f = random.randint(1, 10)
            if f == 1:
                im_sign = Image.open(jpgdir + "/" + sign_t[obj_p["category"]])
                if add_artificial(img, str(cropped_image_count), obj_p, im_sign, obj_p["category"]) == 1:
                    cropped_image_count += 1
                    counter_v2[obj_p["category"]] = counter_v2[obj_p["category"]] + 1
            else:
                if add_data(img, str(cropped_image_count), obj_p, obj_p["category"]) == 1:
                    cropped_image_count += 1
                    counter_v2[obj_p["category"]] = counter_v2[obj_p["category"]] + 1
    if counter_v2["pl30"] == max_data and counter_v2["pl40"] == max_data and counter_v2["pl50"] == max_data\
            and counter_v2["pl60"] == max_data and counter_v2["pl80"] == max_data:
        break
print(counter_v2)
print("loss_data_count")
print(loss_data_count)
f_label.close()
f_log.close()
for key in counter:
    if counter[key] > 500:
        print("{}, {}".format(key, counter[key]))


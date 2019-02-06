import cv2
import anno_func
import json
import math
import random
import numpy as np
from PIL import Image
import os


def cul_id(pl, count):
    if pl == "pl30":
        sign_id = count % 4 + 1
        label_id = sign_id - 1
    elif pl == "pl40":
        sign_id = (count % 4) + 1
        label_id = sign_id + 2
        if sign_id < 2:
            sign_id = 0
            label_id = 0
    elif pl == "pl50":
        sign_id = count % 4 + 1
        label_id = sign_id + 4
        if sign_id < 3:  # 1,2
            sign_id = sign_id - 1
            label_id = sign_id * 3 + 1
    elif pl == "pl60":
        sign_id = count % 4 + 1
        label_id = 9
        if sign_id < 4:  # 1, 2, 3
            sign_id = sign_id - 1
            if sign_id == 2:
                label_id = 7
            elif sign_id == 1:
                label_id = 5
            elif sign_id == 0:
                label_id = 2
    else:
        sign_id = count % 4
        label_id = 9
        if sign_id == 2:
            label_id = 8
        elif sign_id == 1:
            label_id = 6
        elif sign_id == 0:
            label_id = 3
    return sign_id, label_id


def add_data(img_o, out_id, obj_pp, img_sign, label):
    size_base = obj_pp["bbox"]["xmax"] - obj_pp["bbox"]["xmin"]
    place_base = np.array([(obj_pp["bbox"]["xmax"] + obj_pp["bbox"]["xmin"]) / 2,
                           (obj_pp["bbox"]["ymax"] + obj_pp["bbox"]["ymin"]) / 2])
    for i in range(4000):

        size = np.random.randint(min_size, max_size, (2, 1))
        position = (1 - size / 256) * np.random.rand(2, 2)
        center = position + size / 256 / 2
        test = np.abs(center[0] - center[1]) - np.sum(size / 112) / 2
        t = np.sum(np.sign(test))
        if t <= -2:
            continue
        size_after = size[0]
        place_after = 256 * position[0] + (size_after / 2) * np.ones(2)
#         place_after = np.random.rand(2) * (256 - size_after - 32) + (16 + size_after / 2) * np.ones(2)
        size_crop = 256 * size_base / size_after
        place_crop = place_base - place_after * size_base / size_after
        # check crop area
        if place_crop[0] < 0 or place_crop[1] < 0 \
                or place_crop[0] + size_crop > img_o.size[0] or place_crop[1] + size_crop > img_o.size[1]:
            continue
        else:
            img_o = img_o.crop((int(place_crop[0]), int(place_crop[1]), int(place_crop[0] + size_crop)
                               , int(place_crop[1] + size_crop)))
            img_o = img_o.resize((256, 256), Image.LANCZOS)

            expand = 0.7 + 0.6 * random.random()
            a_p = (int(position[1][0] * 256), int(position[1][1] * 256))
            sign = img_sign.resize((size[1], int(size[1] * expand)))
            mask = sign.split()[3]
            img_o.paste(sign, a_p, mask)

            img_o.save(savedir + str(out_id) + ".jpg")
            f_log.write(str(out_id) + ".jpg, " + obj_pp["category"] + "\n")
            f_log.write(str(obj_pp) + "\n")
            f_log.write("place_base" + str(place_base) + "\n")
            f_log.write("size_after" + str(size_after) + "\n")
            f_log.write("size_crop" + str(size_crop) + "\n")
            f_log.write("place_crop" + str(place_crop) + "\n")

            # label file format
            # image_path, label
            f_label.write(str(out_id) + ".jpg, " + str(label) + "\n")
            if str(out_id).startswith("test"):
                f_test.write(str(out_id) + ".jpg, " + str(label) + "\n")
            else:
                f_train.write(str(out_id) + ".jpg, " + str(label) + "\n")
            return 1
    return 0


# http://cg.cs.tsinghua.edu.cn/traffic-sign/
datadir = "C:/Users/waka-lab/Documents/data/data/"
base = os.path.dirname(os.path.abspath(__file__))
jpgdir = os.path.normpath(os.path.join(base, './dataset/sign/'))

json_fp = open('annotations.json')
jf = json.loads(json_fp.read())

# output dir
savedir = datadir + "easymult/"
f_log = open(savedir + 'log.txt', 'w')
f_label = open(savedir + 'label.txt', 'w')
f_test = open(savedir + 'test.txt', 'w')
f_train = open(savedir + 'train.txt', 'w')

min_size = 64
max_size = 192
# if debug is true , stop making data at 100
debug = False
# label
# (3,4) 1 (3,5) 2
#
sign_list = ["30.png", "40.png", "50.png", "60.png", "80.png"]
counter_v2 = {"pl30": 0, "pl40": 0, "pl50": 0, "pl60": 0, "pl80": 0, "pl100": 0}
counter = {"sum": 0}
for key in jf['types']:
    counter[key] = 0

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

    if img_pl_counter == 1 and counter_v2[obj_p["category"]] < 100:
        sign_i, label_i = cul_id(obj_p["category"], counter_v2[obj_p["category"]])
        i_sign = Image.open(jpgdir + "/" + sign_list[sign_i])
        img = Image.open(datadir + path)
        if add_data(img, "test" + str(cropped_image_count), obj_p, i_sign, label_i) == 1:
            cropped_image_count += 1
            counter_v2[obj_p["category"]] = counter_v2[obj_p["category"]] + 1
    else:
        train_list.append(img_id)
cropped_image_count = 0
counter_v2 = {"pl30": 0, "pl40": 0, "pl50": 0, "pl60": 0, "pl80": 0, "pl100": 0}
for inf in range(20):
    if debug:
        break
    for img_id in train_list:
        img_an = jf['imgs'].get(img_id)
        path = img_an['path']
        img_pl_counter = 0
        obj_p = ""
        # judge if there are only one pl traffic sign ,make data
        for img_obj in img_an['objects']:
            obj = img_obj["category"]
            if obj == "pl30" or obj == "pl40" or obj == "pl50" or obj == "pl60" or obj == "pl80":
                img_pl_counter += 1
                obj_p = img_obj
        if img_pl_counter == 1 and counter_v2[obj_p["category"]] < 2000:
            sign_i, label_i = cul_id(obj_p["category"], counter_v2[obj_p["category"]])
            i_sign = Image.open(jpgdir + "/" + sign_list[sign_i])
            img = Image.open(datadir + path)
            if add_data(img, cropped_image_count, obj_p, i_sign, label_i) == 1:
                cropped_image_count += 1
                counter_v2[obj_p["category"]] = counter_v2[obj_p["category"]] + 1
    if counter_v2["pl30"] == 2000 and counter_v2["pl40"] == 2000 and counter_v2["pl50"] == 2000 and counter_v2["pl60"]\
            == 2000 and counter_v2["pl80"] == 2000:
        break
print(counter_v2)
print("loss_data_count")
print(loss_data_count)
f_label.close()
f_log.close()
f_test.close()
f_train.close()

for key in counter:
    if counter[key] > 500:
        print("{}, {}".format(key, counter[key]))


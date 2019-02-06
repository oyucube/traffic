import cv2
import anno_func
import json
import math
import random
import numpy as np
from PIL import Image


def add_data(img_o, out_id, obj_pp):
    size_base = obj_pp["bbox"]["xmax"] - obj_pp["bbox"]["xmin"]
    place_base = np.array([(obj_pp["bbox"]["xmax"] + obj_pp["bbox"]["xmin"]) / 2,
                           (obj_pp["bbox"]["ymax"] + obj_pp["bbox"]["ymin"]) / 2])
    for i in range(10):
        size_after = int(48)
        place_after = 128 * np.ones(2)
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
            img_o.save(savedir + str(out_id) + ".jpg")
            f_log.write(str(out_id) + ".jpg, " + obj_pp["category"] + "\n")
            f_log.write(str(obj_pp) + "\n")
            f_log.write("place_base" + str(place_base) + "\n")
            f_log.write("size_after" + str(size_after) + "\n")
            f_log.write("size_crop" + str(size_crop) + "\n")
            f_log.write("place_crop" + str(place_crop) + "\n")
            # label file format
            # image_path, label
            f_label.write(str(out_id) + ".jpg, " + obj_pp["category"] + "\n")
            return 1
    return 0
# http://cg.cs.tsinghua.edu.cn/traffic-sign/
# %matplotlib inline
# source data dir
datadir = "C:/Users/waka-lab/Documents/data/data/"
json_fp = open('annotations.json')
jf = json.loads(json_fp.read())

# output dir
savedir = datadir + "reward48/"
f_log = open(savedir + 'log.txt', 'w')
f_label = open(savedir + 'label.txt', 'w')
############
# config
############
# newdata
# min_size = 64
# max_size = 128

min_size = 64
max_size = 128
# if debug is true , stop making data at 100
debug = True

counter_v2 = {"pl30": 0, "pl40": 0, "pl50": 0, "pl60": 0, "pl80": 0, "pl100": 0}
counter = {"sum": 0}
for key in jf['types']:
    counter[key] = 0

loss_data_count =0
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
        if add_data(img, "test" + str(cropped_image_count), obj_p) == 1:
            cropped_image_count += 1
            counter_v2[obj_p["category"]] = counter_v2[obj_p["category"]] + 1
    else:
         train_list.append(img_id)
# cropped_image_count = 0
# for inf in range(4):
#     for img_id in train_list:
#         img_an = jf['imgs'].get(img_id)
#         path = img_an['path']
#         img_pl_counter = 0
#         obj_p = ""
#     # if there are only one pl traffic sign ,make data
#         for img_obj in img_an['objects']:
#             obj = img_obj["category"]
#             if obj == "pl30" or obj == "pl40" or obj == "pl50" or obj == "pl60" or obj == "pl80":
#                 img_pl_counter += 1
#                 obj_p = img_obj
#
#         if img_pl_counter == 1 and counter_v2[obj_p["category"]] < 1000:
#             if not(obj_p["category"] == "pl30" or obj_p["category"] == "pl40" or obj_p["category"] == "pl50"
#                     or obj_p["category"] == "pl60" or obj_p["category"] == "pl80" or obj_p["category"] == "pl100"):
#                 print("error")
#                 print(path)
#             img = Image.open(datadir + path)
#             if add_data(img, cropped_image_count, obj_p) == 1:
#                 cropped_image_count += 1
#                 counter_v2[obj_p["category"]] = counter_v2[obj_p["category"]] + 1
#     if counter_v2["pl30"] == 1000 and counter_v2["pl40"] == 1000 and counter_v2["pl50"] == 1000 and counter_v2["pl60"]\
#             == 1000 and counter_v2["pl80"] == 1000:
#         break
# print(counter_v2)
print("loss_data_count")
print(loss_data_count)
f_label.close()
f_log.close()
for key in counter:
    if counter[key] > 500:
        print("{}, {}".format(key, counter[key]))


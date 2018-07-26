# http://cg.cs.tsinghua.edu.cn/traffic-sign/
import cv2
import anno_func
import json
json_fp = open('annotations.json')
jf = json.loads(json_fp.read())

",".join(jf['types'])

counter = {"sum": 0}

for key in jf['types']:
    counter[key] = 0
print(counter)

for img_id in jf['imgs']:
    # print(img_id)
    img_an = jf['imgs'].get(img_id)
#     print(img_an)
    for img_obj in img_an['objects']:
        # print(img_obj["category"])
        obj = img_obj["category"]
        if obj == "pl30" or obj == "pl40" or obj == "pl50" or obj == "pl60" or obj == "pl80" or obj == "pl100":
            print(img_id)
            counter[obj] = counter[obj] + 1
            break
print(counter)

for key in counter:
    if counter[key] > 300:
        print("{}, {}".format(key, counter[key]))


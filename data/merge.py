#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13.06.2024 20:27
# @Author  : Chengjie
# @File    : merge.py
# @Software: PyCharm
import json
import os
import random
from collections import Counter


def get_ann_coo(images_path='test'):
    images = os.listdir(os.path.join(images_path))
    images = [images_path + '/' + image for image in images]
    images = [file for file in images if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    with open(images_path + '/annotations.coco.json') as f:
        ann_coco = json.load(f)

    image_ids = []
    for image_n in images:
        image_name = image_n.split('/')[-1]
        image_id = [ann_coco['images'][i]['id'] for i in range(len(ann_coco['images'])) if
                    ann_coco['images'][i]['file_name'] == image_name]

        if len(image_id) != 1:
            # print(image_name)
            continue
        # assert len(image_id) == 1

        image_ids.append(image_id[0])

    return image_ids, ann_coco


def get_anno(ann_coco, i_id):
    annotations = [annotation for annotation in ann_coco['annotations'] if
                   annotation['image_id'] == i_id]
    image = [image for image in ann_coco['images'] if
             image['id'] == i_id]
    # print(image, annotations)
    return annotations, image


def get_new_anno(ann_coco, i_id, i_id_new):
    annotations = [annotation for annotation in ann_coco['annotations'] if
                   annotation['image_id'] == i_id]
    image = [image for image in ann_coco['images'] if
             image['id'] == i_id]

    # print(image, annotations)
    for ig in image:
        ig['id'] = i_id_new

    for ann in annotations:
        ann['image_id'] = i_id_new
    # print(image, annotations)
    return annotations, image


def append_anno(anno_pre, anno_new):
    anno_ids = [a['id'] for a in anno_pre]
    for a in anno_new:
        if a['id'] not in anno_ids:
            anno_pre.append(a)
            anno_ids.append(a['id'])
        else:
            id_ = random.randint(1, 1000)
            while id_ in anno_ids:
                id_ = random.randint(1, 1000)
            a['id'] = id_
            anno_pre.append(a)
            anno_ids.append(a['id'])
    return anno_pre


image_ids_test, ann_coco_test = get_ann_coo(images_path='test')
image_ids_val, ann_coco_val = get_ann_coo(images_path='val')

for image_i in image_ids_test:
    if image_i not in image_ids_val:
        anno, img = get_anno(ann_coco_test, image_i)

        ann_coco_val['images'] = ann_coco_val['images'] + img
        # ann_coco_val['annotations'] = ann_coco_val['annotations'] + anno
        ann_coco_val['annotations'] = append_anno(ann_coco_val['annotations'], anno)
        image_ids_val.append(image_i)

    else:
        i_new = random.randint(500, 600)
        while i_new in image_ids_val:
            i_new = random.randint(500, 600)
        anno, img = get_new_anno(ann_coco_test, image_i, i_new)

        ann_coco_val['images'] = ann_coco_val['images'] + img
        # ann_coco_val['annotations'] = ann_coco_val['annotations'] + anno
        ann_coco_val['annotations'] = append_anno(ann_coco_val['annotations'], anno)
        image_ids_val.append(i_new)

print(len(ann_coco_val['images']))
with open('val/annotations.coco.json', 'w') as f:
    json.dump(ann_coco_val, f, indent=4)

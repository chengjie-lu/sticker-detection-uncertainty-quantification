#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/05/2024 01:00
# @Author  : Chengjie
# @File    : resize_image.py
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt


def resize_sticker(f_n, n_fn):
    img = cv2.imread(f_n)
    resized_image = cv2.resize(img, (2448, 2048))
    cv2.imwrite(n_fn, resized_image)


if __name__ == '__main__':
    file_n = './noisy_image_open43.jpg'
    new_fn = './noisy_image_open43.jpg'

    resize_sticker(f_n=file_n, n_fn=new_fn)

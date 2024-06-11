#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/05/2024 01:00
# @Author  : Chengjie
# @File    : resize_image.py
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np


def resize_sticker(f_n, n_fn):
    img = cv2.imread(f_n)
    resized_image = cv2.resize(img, (2448, 2048))
    # resized_image = img[:, 0:2448]
    cv2.imwrite(n_fn, resized_image)


def crop_sticker(f_n, n_fn):
    img = cv2.imread(f_n)
    # resized_image = cv2.resize(img, (2448, 2048))
    resized_image = img[:, 0:2448]
    cv2.imwrite(n_fn, resized_image)


def image_difference():
    image1 = cv2.imread('12_resized.jpg')
    image2 = cv2.imread('image_top54.jpg')

    if image1.shape != image2.shape:
        raise ValueError("Images must be the same size for subtraction")

    result = cv2.subtract(image1, image2)

    # Display the result
    cv2.imshow('Subtracted Image', result * 200)

    print(result)
    # Save the result
    cv2.imwrite('subtracted_image.jpg', result * 200)


if __name__ == '__main__':
    file_n = 'mac_top.jpg'
    new_fn = 'mac_top_resize.jpg'
    #
    resize_sticker(f_n=file_n, n_fn=new_fn)

    # image_difference()

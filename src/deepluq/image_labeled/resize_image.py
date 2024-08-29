#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/05/2024 01:00
# @Author  : Chengjie
# @File    : resize_image.py
# @Software: PyCharm
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def resize_sticker(f_n, n_fn):
    img = cv2.imread(f_n)
    resized_image = cv2.resize(img, (2448, 2048))
    cv2.imwrite(n_fn, resized_image)


def crop_sticker(f_n, n_fn):
    img = cv2.imread(f_n)
    resized_image = img[300:1808, 200:2250]
    cv2.imwrite(n_fn, resized_image * 2)


def image_difference(i_1, i_2, i):
    image1 = cv2.imread(i_1)
    image2 = cv2.imread(i_2)

    if image1.shape != image2.shape:
        raise ValueError("Images must be the same size for subtraction")

    result = cv2.subtract(image1, image2)

    # Display the result
    # cv2.imshow('Subtracted Image', result * 200)

    # print(np.sum(result))
    # Save the result
    cv2.imwrite('subtracted_image_{}.jpg'.format(i), result * 100)


if __name__ == '__main__':
    file_n = '../test_images/subtracted_image_97.jpg'
    # new_fn = 'image_stable_diffusion_97_per.jpg'
    # image_difference(file_n, new_fn, 97)
    crop_sticker(file_n, 'diff_97.jpg')

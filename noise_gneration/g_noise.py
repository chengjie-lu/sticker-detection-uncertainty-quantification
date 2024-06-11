#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/05/2024 13:14
# @Author  : Chengjie
# @File    : g_noise.py
# @Software: PyCharm

import numpy as np
from PIL import Image

# Load the original image
image_path = "../test_images/image_open45.jpg"  # Replace with the path to your image
image = Image.open(image_path)
image_array = np.array(image)


# Add small perturbations (Gaussian noise with low standard deviation)
def add_small_perturbations(image_array, mean=0, std=2):
    gaussian_noise = np.random.normal(mean, std, image_array.shape).astype('int16')
    perturbed_image = image_array + gaussian_noise
    perturbed_image = np.clip(perturbed_image, 0, 255).astype('uint8')
    return perturbed_image


# Apply small perturbations
perturbed_image_array = add_small_perturbations(image_array)
perturbed_image = Image.fromarray(perturbed_image_array)

# Save the perturbed image
perturbed_image_path = "path_to_save_perturbed_image.jpg"  # Replace with your desired output path
perturbed_image.save(perturbed_image_path)

print(f"Perturbed image saved at {perturbed_image_path}")

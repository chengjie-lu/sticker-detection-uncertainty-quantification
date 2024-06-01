#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/05/2024 14:22
# @Author  : Chengjie
# @File    : stable_difussion.py
# @Software: PyCharm

import getpass

import requests

STABILITY_KEY = getpass.getpass('Enter your API Key')


# sk-6mfIdTkpqQSLLxZczdez6hTVRxE3BHCH8CK3UIF1L1qLuCpY

def send_generation_request(
        host,
        params,
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response


# @title SD3
# @markdown - Drag and drop image to file folder on left
# @markdown - Right click it and choose Copy path
# @markdown - Paste that path into image field below
# @markdown <br><br>

image = "../data/test/image_open43.jpg"  # @param {type:"string"}
prompt = "Add perturbations on this image"  # @param {type:"string"}
negative_prompt = ""  # @param {type:"string"}
seed = 0  # @param {type:"integer"}
output_format = "jpeg"  # @param ["jpeg", "png"]
strength = 0.75  # @param {type:"slider", min:0.0, max: 1.0, step: 0.01}

host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"

params = {
    "image": image,
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "strength": strength,
    "seed": seed,
    "output_format": output_format,
    "mode": "image-to-image"
}

response = send_generation_request(
    host,
    params
)

# Decode response
output_image = response.content
finish_reason = response.headers.get("finish-reason")
seed = response.headers.get("seed")

# Check for NSFW classification
if finish_reason == 'CONTENT_FILTERED':
    raise Warning("Generation failed NSFW classifier")

# Save and display result
generated = f"generated_{seed}.{output_format}"
with open(generated, "wb") as f:
    f.write(output_image)
print(f"Saved image {generated}")

# output.no_vertical_scroll()
# print("Result image:")
# IPython.display.display(Image.open(generated))

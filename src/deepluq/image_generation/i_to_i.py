#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/05/2024 01:23
# @Author  : Chengjie
# @File    : i-to-i.py
# @Software: PyCharm

import base64
import os
import requests

engine_id = "stable-diffusion-v1-6"
api_host = os.getenv("API_HOST", "https://api.stability.ai")
# api_key = os.getenv("STABILITY_API_KEY")
#
# if api_key is None:
#     raise Exception("Missing Stability API key.")

response = requests.post(
    f"{api_host}/v1/generation/{engine_id}/image-to-image",
    headers={
        "Accept": "application/json",
        "Authorization": f"Bearer sk-6mfIdTkpqQSLLxZczdez6hTVRxE3BHCH8CK3UIF1L1qLuCpY",
    },
    files={"init_image": open("../test_images/vision_model/laptop.jpeg", "rb")},
    data={
        "image_strength": 0.35,
        "init_image_mode": "IMAGE_STRENGTH",
        "text_prompts[0][text]": "increase the image brightness",
        "cfg_scale": 7,
        "samples": 1,
        "steps": 30,
    },
)

if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))

data = response.json()

for i, image in enumerate(data["artifacts"]):
    with open(f"./image_45_{i}.png", "wb") as f:
        f.write(base64.b64decode(image["base64"]))

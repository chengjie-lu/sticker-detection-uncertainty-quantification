#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/05/2024 01:17
# @Author  : Chengjie
# @File    : stable-diffusion.py
# @Software: PyCharm

import requests

response = requests.post(
    f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
    headers={
        "authorization": f"Bearer xxx",
        "accept": "image/*",
    },
    files={"none": ""},
    data={
        "prompt": "A top-down view of a laptop with two stickers on it. The laptop is lying flat on a surface with its screen open to a 180-degree angle, making it flush with the surface. Both the keyboard and the screen are parallel to the surface, creating a continuous flat layout. This positioning allows both the screen and keyboard to touch the surface simultaneously, giving a clear view of both components from a top-down perspective.",
        # "prompt": 'A top-down view of a Macbook Pro with two stickers on the lid. The laptop is closed with the screen turned off. The stickers include a colorful cartoon character, a geometric pattern, and a nature-themed sticker with a leaf or tree design. The laptop is on a textured surface.',
        "output_format": "jpeg",
    },
)

if response.status_code == 200:
    with open("laptop.jpeg", "wb") as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))

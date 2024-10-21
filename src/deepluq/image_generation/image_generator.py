#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/05/2024 14:05
# @Author  : Chengjie
# @File    : image_generator.py
# @Software: PyCharm
import requests
from openai import OpenAI
import urllib.request


class ImageGenerator:
    def __init__(self):
        self.i_path = None
        self.model = None

    def text_to_image(self, model, prompt, i_path, i_n):
        self.model = model
        self.i_path = i_path
        if self.model == "stable_diffusion":
            api_key = "xxx"
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
                files={"none": ""},
                data={
                    "prompt": prompt,
                    # "prompt": 'A top-down view of a Macbook Pro with two stickers on the lid. The laptop is closed with the screen turned off. The stickers include a colorful cartoon character, a geometric pattern, and a nature-themed sticker with a leaf or tree design. The laptop is on a textured surface.',
                    "output_format": "jpeg",
                },
            )

            # print(response.content)

            if response.status_code == 200:
                with open(i_path + "/{}_{}.jpeg".format(i_n, model), "wb") as file:
                    file.write(response.content)
            else:
                raise Exception(str(response.json()))

        elif model == "dall":
            client = OpenAI(
                api_key="xxx"
            )

            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            image_url = response.data[0].url

            # print(image_url)

            urllib.request.urlretrieve(
                image_url, i_path + "/{}_{}.jpeg".format(i_n, model)
            )

    def image_to_image(self, model):
        self.model = model


if __name__ == "__main__":
    i_g = ImageGenerator()
    p1 = "A top-down view of a Macbook Pro with two stickers on the lid. The laptop is closed with the screen turned off. The stickers include a colorful cartoon character, a geometric pattern, and a nature-themed sticker with a leaf or tree design. The laptop is on a textured surface."
    # p2 = 'A Lenovo ThinkPad X1 Carbon laptop lying open on a flat surface, showing the keyboard and screen. The keyboard has a few colorful stickers, including one with a cartoon character and another with a decorative design. The laptop is positioned with the screen to the left and the keyboard to the right. The background is a plain surface with some screws visible near the edges.'
    # p3 = "A Lenovo ThinkPad X1 Carbon laptop lying open on a plain, slightly textured surface. The screen is on the left, and the keyboard is on the right. The laptop keyboard has a few simple, colorful stickers on the wrist rest area and some keys. The background shows a plain, possibly fabric surface with a couple of screws visible near the edges. The overall setting is minimalistic and utilitarian, highlighting the ThinkPad's design and the stickers on it."
    i_g.text_to_image(model="dall", prompt=p1, i_path="images", i_n="laptop_p1")

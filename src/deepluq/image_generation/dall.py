#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/05/2024 23:23
# @Author  : Chengjie
# @File    : dall.py
# @Software: PyCharm

from openai import OpenAI

client = OpenAI(api_key="sk-proj-GJlWQit9KLcIfzmtZlk5T3BlbkFJTP0mku2G4fITeprFZK7J")

# response = client.images.create_variation(
#     model="dall-e-2",
#     image=open("../test_images/image_open43 Large.png", "rb"),
#     n=1,
#     size="1024x1024"
# )
#
# image_url = response.data[0].url
# print(response)
# print(image_url)

# client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="A top-down view of a Lenovo ThinkPad laptop with a few stickers on the keyboard area. The laptop is open with the screen turned off. The stickers include a colorful cartoon character, a geometric pattern, and a nature-themed sticker with a leaf or tree design. The laptop is on a textured surface.",
    size="1792x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url

print(image_url)

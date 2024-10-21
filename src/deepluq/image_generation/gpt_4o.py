#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/05/2024 22:09
# @Author  : Chengjie
# @File    : gpt-4o.py
# @Software: PyCharm
import os

# from openai import OpenAI
#
# # os.environ.get("sk-proj-GJlWQit9KLcIfzmtZlk5T3BlbkFJTP0mku2G4fITeprFZK7J")
# # client = OpenAI()
# client = OpenAI(api_key="sk-proj-GJlWQit9KLcIfzmtZlk5T3BlbkFJTP0mku2G4fITeprFZK7J")
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What’s in this image?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#                     },
#                 },
#             ],
#         }
#     ],
#     max_tokens=300,
# )
#
# print(response.choices[0].message)

import base64
import requests

# OpenAI API Key
api_key = "xxx"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "../test_images/image_open45.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {api_key}"
# }
#
# payload = {
#     "model": "gpt-4o",
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "how many stickers are on the laptop and can you mark them?"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}"
#                     }
#                 }
#             ]
#         }
#     ],
#     "max_tokens": 300
# }
#
# response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
#
# print(response.json())


from openai import OpenAI

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "generate python code to add noises on it."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        # "detail": "high"
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response)

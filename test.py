#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 01.03.2024 13:29
# @Author  : Chengjie
# @File    : test.py
# @Software: PyCharm
import torch
from torch import nn

# m = nn.Dropout(p=0.2)
# input = torch.randn(20, 16)
# print(input)
# output = m(input)
# print(output)

# a = torch.tensor([1.4, 2.5, 3.1, 4.6])
# print(a.int())
# # keep = [True, False, True]
# # print(a[keep])
# keep = torch.tensor([0, 1])
# b = torch.where(torch.tensor(keep))
# print(b)


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

a = np.array([1663.7069, 1603.4716, 1763.8948, 1696.7184])
b = np.array([1662.7069, 1601.4716, 1761.8948, 1691.7184])

# c = np.append([], a, axis=0)
# print(c)
# d = np.append(c, b, axis=0)
# print(d)

a = a.reshape(1, -1)
b = b.reshape(1, -1)

similarity = cosine_similarity(a, b)[0]
print(similarity)

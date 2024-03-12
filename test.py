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

a = torch.tensor([1.4, 2.5, 3.1, 4.6])
print(a.int())
# keep = [True, False, True]
# print(a[keep])
keep = torch.tensor([0, 1])
b = torch.where(torch.tensor(keep))
print(b)

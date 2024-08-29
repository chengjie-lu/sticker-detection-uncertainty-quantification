#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20.06.2024 11:26
# @Author  : 
# @File    : test.py
# @Software: PyCharm
import os
import time
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR

from sklearn.cluster import DBSCAN
import numpy as np

# X = np.array([[1, 2], [2, 2], [2, 3],
#               [8, 7], [8, 8], [25, 80]])
# clustering = DBSCAN(eps=3, min_samples=2).fit(X)
# clustering.labels_
# clustering

f_n = '/home/complexse/workspace/RoboSapiens/DTI-Laptop-refubishment/sticker_detector/checkpoints/dropout_t.txt'
# os.chmod(f_n, S_IWUSR | S_IREAD)

while int(oct(os.stat(f_n).st_mode)[-3:]) != 600:
    print(time.time())

f = open(f_n, 'w')
f.write(str(3))
f.close()
os.chmod(f_n, S_IREAD)
print(oct(os.stat(f_n).st_mode)[-3:])

time.sleep(10)

f = open(f_n, 'r')
time.sleep(10)
print(f.read())
f.close()
os.chmod(f_n, S_IWUSR | S_IREAD)

print(oct(os.stat(f_n).st_mode)[-3:])

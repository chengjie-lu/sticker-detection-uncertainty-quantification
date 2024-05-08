#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 01.03.2024 13:29
# @Author  : Chengjie
# @File    : test.py
# @Software: PyCharm
import random

import torch
from scipy.spatial import ConvexHull
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

# a = np.array([1663.7069, 1603.4716, 1763.8948, 1696.7184], [1663.7069, 1603.4716, 1763.8948, 1696.7184])
# b = np.array([1662.7069, 1601.4716, 1761.8948, 1691.7184])
#
# print(np.delete(a, 0))
# # c = np.append([], a, axis=0)
# # print(c)
# # d = np.append(c, b, axis=0)
# # print(d)
#
# a = a.reshape(1, -1)
# b = b.reshape(1, -1)
#
# similarity = cosine_similarity(a, b)[0]
# print(similarity)

from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import DBSCAN
import pandas as pd

# X, y = make_classification(
#     n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=1,
#     shuffle=False, random_state=42
# )
#
# print(X, y)
# mutual_info_classif(X, y)

mc_locations = []
for i in range(100):
    mc_locations.append(np.array([random.random(), random.random(), random.random(), random.random()]))

clustering = DBSCAN(eps=100, min_samples=2).fit(mc_locations)

mc_locations = np.c_[mc_locations, clustering.labels_.ravel()]

mc_locations_df = pd.DataFrame(data=mc_locations, columns=['x1', 'y1', 'x2', 'y2', 'label'])

cluster_labels = np.unique(mc_locations[:, 4])
total_cluster_surface = 0.0
avg_surface = -1.0
for cluster_label in cluster_labels:
    print(cluster_label)
    cluster_df = mc_locations_df.query('label == ' + str(cluster_label))
    print(cluster_df.shape)
    if cluster_df.shape[0] > 2:
        center_data = cluster_df[['x1', 'y1']].values
        hull = ConvexHull(center_data)
        total_cluster_surface += hull.area

        center_data = cluster_df[['x2', 'y1']].values
        hull = ConvexHull(center_data)
        total_cluster_surface += hull.area

        center_data = cluster_df[['x1', 'y2']].values
        hull = ConvexHull(center_data)
        total_cluster_surface += hull.area

        center_data = cluster_df[['x2', 'y2']].values
        hull = ConvexHull(center_data)
        total_cluster_surface += hull.area
    avg_surface = total_cluster_surface / mc_locations.shape[0]

print(avg_surface)

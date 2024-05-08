#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 07.05.2024 11:38
# @Author  : Chengjie
# @File    : utils.py
# @Software: PyCharm
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


class DBSCANCluster:
    def __init__(self, x):
        self.cluster = DBSCAN(eps=100, min_samples=2).fit(x)
        self.mc_locations = np.c_[x, self.cluster.labels_.ravel()]
        self.mc_locations_df = pd.DataFrame(data=self.mc_locations, columns=['x1', 'y1', 'x2', 'y2', 'label'])
        self.cluster_labels = np.unique(self.mc_locations[:, len(self.mc_locations[0]) - 1])

    def cluster_preds(self, preds):
        pred_id = 0
        preds_new = {}

        total_cluster_surface = 0.0
        avg_surface = -1.0
        for cluster_label in self.cluster_labels:
            cluster_df = self.mc_locations_df.query('label == ' + str(cluster_label))
            boxs = cluster_df[['x1', 'y1', 'x2', 'y2']].values
            t = 0
            for key in preds.keys():
                if preds[key]['box'] in boxs and t == 0:
                    preds_new.update({'label_{}'.format(pred_id):
                        {
                            'box': [preds[key]['box']],
                            'label': [preds[key]['label']],
                            'score': [preds[key]['score']],
                            'logit': [preds[key]['logit']],
                            'center_point': [preds[key]['center_point']]
                        }
                    })
                    t = 1
                elif preds[key]['box'] in boxs and t != 0:
                    preds_new['label_{}'.format(pred_id)]['box'].append(preds[key]['box'])
                    preds_new['label_{}'.format(pred_id)]['label'].append(preds[key]['label'])
                    preds_new['label_{}'.format(pred_id)]['score'].append(preds[key]['score'])
                    preds_new['label_{}'.format(pred_id)]['logit'].append(preds[key]['logit'])
                    preds_new['label_{}'.format(pred_id)]['center_point'].append(preds[key]['center_point'])

            pred_id += 1

        return preds_new


def cluster(mc_locations):
    clustering = DBSCAN(eps=100, min_samples=2).fit(mc_locations)
    mc_locations = np.c_[mc_locations, clustering.labels_.ravel()]

    mc_locations_df = pd.DataFrame(data=mc_locations, columns=['x1', 'y1', 'x2', 'y2', 'label'])

    cluster_labels = np.unique(mc_locations[:, len(mc_locations[0]) - 1])
    total_cluster_surface = 0.0
    avg_surface = -1.0
    for cluster_label in cluster_labels:
        sf_tmp = 0
        cluster_df = mc_locations_df.query('label == ' + str(cluster_label))
        if cluster_df.shape[0] > 2:
            center_data = cluster_df[['x1', 'y1']].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area

            center_data = cluster_df[['x2', 'y1']].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area

            center_data = cluster_df[['x1', 'y2']].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area

            center_data = cluster_df[['x2', 'y2']].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area
        # print(sf_tmp)
        avg_surface = total_cluster_surface / mc_locations.shape[0]

    # print(total_cluster_surface, avg_surface)

# def dbscan_cluster(mc_locations):

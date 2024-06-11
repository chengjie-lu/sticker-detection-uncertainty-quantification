#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.01.2024 15:21
# @Author  : Chengjie
# @File    : detection_offline.py
# @Software: PyCharm
import json
import time

import cv2 as cv
import pandas as pd
import torch
from sklearn.cluster import DBSCAN

from uq.metrics import UQMetrics
from utils import load_camera_calibration, load_model, run_model, calc_3d_point
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from uq.utils import cluster, DBSCANCluster

RUNTIME_TYPE = 'normal'  # Choices == 'onnx' and 'normal'
LABELS = {'Background': 0, 'Logo': 1, 'Sticker': 2}


class Detection:
    def __init__(self):
        self.model = load_model(RUNTIME_TYPE)
        self.p, self.d, self.dist_maps = load_camera_calibration()
        self.min_score = 0.6
        self.show_bounding_box_sticker = True
        self.show_bounding_box_logo = True
        self.uq = UQMetrics()
        # self.image_resize_variable = tk.IntVar()
        # self.image_resize_variable.set(100)

    def process_images(self, path="test_images/IMG_3774.jpg"):
        image_og = cv.imread(path)
        # Undistorts images with the maps calculated in load_camera_calibration()
        image_og = cv.remap(image_og, self.dist_maps[0], self.dist_maps[1], cv.INTER_LINEAR)
        # image_og = cv.undistort(image_og, self.p, self.d)
        # scale = self.image_resize_variable.get() / 100.0

        # image_og = cv.rotate(image_og, cv.ROTATE_90_CLOCKWISE)
        scale = 1.0

        image_rz = cv.resize(image_og, (0, 0), fx=scale, fy=scale)
        return image_og, image_rz

    def predict(self, image_rz):
        preds = run_model(image_rz, self.model, RUNTIME_TYPE)

        # print(preds[0]['boxes'].cpu().detach().numpy())
        # print(preds[0]['labels'].cpu().detach().numpy())
        # print(preds[0]['scores'].cpu().detach().numpy())

        for j in range(len(preds[0]['scores']) - 1, -1, -1):
            if preds[0]['scores'][j] < self.min_score or preds[0]['labels'][j] == 0 or (
                    preds[0]['labels'][j] == 1 and not self.show_bounding_box_logo):
                preds[0]['boxes'] = torch.cat((preds[0]['boxes'][:j], preds[0]['boxes'][j + 1:]))
                preds[0]['labels'] = torch.cat((preds[0]['labels'][:j], preds[0]['labels'][j + 1:]))
                preds[0]['scores'] = torch.cat((preds[0]['scores'][:j], preds[0]['scores'][j + 1:]))
                preds[0]['logits'] = torch.cat((preds[0]['logits'][:j], preds[0]['logits'][j + 1:]))
        for pred in preds:
            for key, value in pred.items():
                pred[key] = value.cpu()

        boxes = preds[0]['boxes'].cpu().detach().numpy()
        labels = preds[0]['labels'].cpu().detach().numpy()
        scores = preds[0]['scores'].cpu().detach().numpy()
        logits = preds[0]['logits'].cpu().detach().numpy()

        # print('==============================')
        # print(np.delete(boxes, 0, axis=0))
        # print(scores)
        # print(logits)
        return preds, boxes, labels, scores, logits

    def draw_boxes(self, boxes, labels, scores, image_og):
        scale = 1.0
        boxes = boxes / scale

        center_3d_points = []
        labels = [key for value in labels for key, val in LABELS.items() if val == value]
        for box, label, score in zip(boxes, labels, scores):
            if self.show_bounding_box_sticker:
                if label == 'Sticker':
                    image_og = cv.rectangle(image_og, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                            (255, 0, 0), 2)

                    # image_og = cv.putText(image_og, str(label + ' ' + str(score)[:5]), (int(box[0]), int(box[1])),
                    #                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                    #                       cv.LINE_AA)  # (255,0,0) is blue
                    image_og = cv.putText(image_og, str(label), (int(box[0]), int(box[1])),
                                          cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                                          cv.LINE_AA)  # (255,0,0) is blue

                    x = int((box[0] + box[2]) / 2)
                    y = int((box[1] + box[3]) / 2)
                    center_3d_str, center_3d = calc_3d_point(box, self.p)
                    center_3d_points.append(center_3d)
                    # draw
                    image_og = cv.circle(image_og, (int(x), int(y)), 3, (0, 0, 255), -1)
                    # image_og = cv.putText(image_og, center_3d_str, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 2,
                    #                       (0, 0, 255), 2, cv.LINE_AA)

            if self.show_bounding_box_logo:
                if label == 'Logo':
                    image_og = cv.rectangle(image_og, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                            (255, 0, 0), 2)
                    # image_og = cv.putText(image_og, str(label + ' ' + str(score)[:5]), (int(box[0]), int(box[1])),
                    #                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                    #                       cv.LINE_AA)  # (255,0,0) is blue
                    image_og = cv.putText(image_og, str(label), (int(box[0]), int(box[1])),
                                          cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                                          cv.LINE_AA)  # (255,0,0) is blue

                    x = int((box[0] + box[2]) / 2)
                    y = int((box[1] + box[3]) / 2)
                    center_3d_str, center_3d = calc_3d_point(box, self.p)
                    center_3d_points.append(center_3d)
                    # draw
                    image_og = cv.circle(image_og, (int(x), int(y)), 3, (0, 0, 255), -1)
                    # image_og = cv.putText(image_og, center_3d_str, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 2,
                    #                       (0, 0, 255), 2, cv.LINE_AA)

        # cv.imwrite('./image_labeled/test_{}.png'.format(time.time()), image_og)

        return center_3d_points

    @staticmethod
    def put_text(text, box, image):
        cv.putText(image, text, (int(box[0] - 260), int(box[1])),
                   cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                   cv.LINE_AA)  # (255,0,0) is blue

    def predict_multi(self, image_rz, image_og, image_n, n=10):
        predictions = {}
        pred_id = 0

        mc_locations = []

        for i in range(n):
            preds, boxes, labels, scores, logits = self.predict(image_rz)
            center_points = self.draw_boxes(boxes, labels, scores, image_og)

            for box, label, score, logit, center_point in zip(boxes, labels, scores, logits, center_points):
                mc_locations.append(box)

                predictions.update({'label_{}'.format(pred_id):
                    {
                        'box': box.tolist(),
                        'label': label.tolist(),
                        'score': score.tolist(),
                        'logit': logit.tolist(),
                        'center_point': center_point
                    }
                })
                pred_id += 1

        dbscan_cluster = DBSCANCluster(x=mc_locations)
        predictions = dbscan_cluster.cluster_preds(preds=predictions)
        # print(json.dumps(predictions, indent=4))

        # cluster(mc_locations)

        for key in predictions.keys():
            self.put_text(key, predictions[key]['box'][0], image_og)
            logit_sample_trans = np.transpose(predictions[key]['logit'])
            shannon_entropy = self.uq.calcu_entropy(np.mean(logit_sample_trans, axis=1))
            # mutual_info = self.uq.calcu_mutual_information(logit_sample_trans[0],
            #                                                logit_sample_trans[1],
            #                                                logit_sample_trans[2])
            mi = self.uq.calcu_mi(predictions[key]['logit'])
            tv_cp = self.uq.calcu_tv(predictions[key]['center_point'], tag='center_point')
            tv_box = self.uq.calcu_tv(predictions[key]['box'], tag='bounding_box')
            predictive_surface = self.uq.calcu_prediction_surface(predictions[key]['box'])

            predictions[key].update({
                'detection times (out of {})'.format(n): len(predictions[key]['score']),
                'entropy [classification]': shannon_entropy,
                'mutual_info [classification]': mi,
                'total_var_center_point [regression]': tv_cp,
                'total_var_box [regression]': tv_box,
                'predictive_surface [regression]': predictive_surface
            })

        # print(json.dumps(predictions, indent=4))
        # timestamp = str(int(time.time()))
        tag = image_n
        with open('./image_labeled/output_{}.json'.format(tag), 'w') as f:
            json.dump(predictions, f, indent=4)
        cv.imwrite('./image_labeled/multi_boxes_{}.png'.format(tag), image_og)

    def predict_multi_draw(self, image_rz, image_og, n=10):
        for i in range(n):
            preds, boxes, labels, scores, logits = self.predict(image_rz)
            self.draw_boxes(boxes, labels, scores, image_og)
        cv.imwrite('./image_labeled/multi_boxes_{}.png'.format(time.time()), image_og)


if __name__ == '__main__':
    i_n = 'z_top'
    detector = Detection()
    i_og, i_rz = detector.process_images(path="test_images/{}.jpg".format(i_n))
    # detector.predict_multi_draw(i_rz, i_og, n=1)
    detector.predict_multi(i_rz, i_og, i_n, n=40)

# score: softmax/sigmoid probability

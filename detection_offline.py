#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.01.2024 15:21
# @Author  : Chengjie
# @File    : detection_offline.py
# @Software: PyCharm
import time

import cv2 as cv
import torch

from uq.metrics import UQMetrics
from utils import load_camera_calibration, load_model, run_model, calc_3d_point
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        # print(boxes)
        # print(scores)
        # print(logits)
        return preds, boxes, labels, scores, logits

    def draw_boxes(self, boxes, labels, scores, image_og):
        scale = 1.0
        boxes = boxes / scale

        labels = [key for value in labels for key, val in LABELS.items() if val == value]
        for box, label, score in zip(boxes, labels, scores):
            if self.show_bounding_box_sticker:
                if label == 'Sticker':
                    image_og = cv.rectangle(image_og, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                            (255, 0, 0), 2)
                    image_og = cv.putText(image_og, str(label + ' ' + str(score)[:5]), (int(box[0]), int(box[1])),
                                          cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                                          cv.LINE_AA)  # (255,0,0) is blue

                    x = int((box[0] + box[2]) / 2)
                    y = int((box[1] + box[3]) / 2)
                    center_3d_str = calc_3d_point(box, self.p)
                    # draw
                    image_og = cv.circle(image_og, (int(x), int(y)), 3, (0, 0, 255), -1)
                    image_og = cv.putText(image_og, center_3d_str, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 2,
                                          (0, 0, 255), 2, cv.LINE_AA)

            if self.show_bounding_box_logo:
                if label == 'Logo':
                    image_og = cv.rectangle(image_og, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                            (255, 0, 0), 2)
                    image_og = cv.putText(image_og, str(label + ' ' + str(score)[:5]), (int(box[0]), int(box[1])),
                                          cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                                          cv.LINE_AA)  # (255,0,0) is blue

                    x = int((box[0] + box[2]) / 2)
                    y = int((box[1] + box[3]) / 2)
                    center_3d_str = calc_3d_point(box, self.p)
                    # draw
                    image_og = cv.circle(image_og, (int(x), int(y)), 3, (0, 0, 255), -1)
                    image_og = cv.putText(image_og, center_3d_str, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 2,
                                          (0, 0, 255), 2, cv.LINE_AA)

        cv.imwrite('./image_labeled/test_{}.png'.format(time.time()), image_og)

    @staticmethod
    def insert(pre_dict, box, label, score, logit, pred_id):
        similarity, insert_key = 0, None
        for key in pre_dict.keys():
            box1 = np.array(pre_dict[key]['box'][0]).reshape(1, -1)
            box2 = box.reshape(1, -1)
            sim = cosine_similarity(box1, box2)[0]
            if similarity < sim:
                similarity = sim
                insert_key = key

        if similarity > 0.99:
            pre_dict[insert_key]['box'].append(box.tolist())
            pre_dict[insert_key]['label'].append(label.tolist())
            pre_dict[insert_key]['score'].append(score.tolist())
            pre_dict[insert_key]['logit'].append(logit.tolist())
        else:
            pre_dict.update({'pred_{}'.format(pred_id):
                {
                    'box': [box.tolist()],
                    'label': [label.tolist()],
                    'score': [score.tolist()],
                    'logit': [logit.tolist()]
                }
            })
            pred_id += 1
        # print(pre_dict)

    def predict_multi(self, image_rz, image_og, n=10):
        predictions = {}
        pred_id = 0
        for i in range(n):
            entropy_i = []
            preds, boxes, labels, scores, logits = self.predict(image_rz)

            for box, label, score, logit in zip(boxes, labels, scores, logits):
                # if predictions == {}:
                if i == 0:
                    print(pred_id)
                    predictions.update({'pred_{}'.format(pred_id):
                        {
                            'box': [box.tolist()],
                            'label': [label.tolist()],
                            'score': [score.tolist()],
                            'logit': [logit.tolist()]
                        }
                    })
                    pred_id += 1
                else:
                    self.insert(predictions, box, label, score, logit, pred_id)

            # print(boxes)
            # for pl in logits:
            #     entropy_i.append(self.uq.entropy(pl))

            self.draw_boxes(boxes, labels, scores, image_og)
        print(predictions)
        cv.imwrite('./image_labeled/multi_boxes_{}.png'.format(time.time()), image_og)

    def predict_multi_draw(self, image_rz, image_og, n=10):
        for i in range(n):
            preds, boxes, labels, scores, logits = self.predict(image_rz)
            self.draw_boxes(boxes, labels, scores, image_og)
        cv.imwrite('./image_labeled/multi_boxes_{}.png'.format(time.time()), image_og)


if __name__ == '__main__':
    detector = Detection()
    i_og, i_rz = detector.process_images(path="test_images/test.jpg")

    # p, b, l, s = detector.predict(i_rz)
    # detector.draw_boxes(b, l, s, i_og)
    detector.predict_multi(i_rz, i_og, n=3)

# score: softmax/sigmoid probability

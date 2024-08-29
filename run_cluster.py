#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 23.06.2024 17:05
# @Author  : Chengjie
# @File    : run_cluster.py.py
# @Software: PyCharm
import argparse
import os

import pandas as pd

from detection_offline_dbscan import run_uq

models = {
    'retinanet_resnet50_fpn': 'checkpoints/retinanet_resnet50_fpn/epoch=30-step=7471.ckpt',
    'retinanet_resnet50_fpn_v2': 'checkpoints/retinanet_resnet50_fpn_v2/epoch=31-step=7712.ckpt',
    'fasterrcnn_resnet50_fpn': 'checkpoints/fasterrcnn_resnet50_fpn/epoch=14-step=1815.ckpt',
    'fasterrcnn_resnet50_fpn_v2': 'checkpoints/fasterrcnn_resnet50_fpn_v2/epoch=16-step=8177.ckpt',
    'ssd300_vgg16': 'checkpoints/ssd300_vgg16/epoch=33-step=4114.ckpt',
    'ssdlite320_mobilenet_v3_large': 'checkpoints/ssdlite320_mobilenet_v3_large/epoch=18-step=9139.ckpt'
}


def main(args):
    print(args.model_n, args.dataset_p, args.dropout)

    # dataset_n = args.dataset_p

    headers = ['image', 'image_name', 'MAP', 'UQ[VR]', 'UQ[IE]', 'UQ[MI]', 'UQ[TR]', 'UQ[PS]',
               'MAP[50]', 'MAP[75]', 'MAP[Small]', 'MAP[Medium]', 'MAP[Large]', 'MAR[1]', 'MAR[5]', 'MAR[100]',
               'MAR[Small]', 'MAR[Medium]', 'MAR[Large]']

    logs = 'experiment_results/{}/dataset/{}'.format(args.model_n, args.dataset_p)
    if not os.path.exists(logs):
        os.makedirs(logs)

    f_n = './experiment_results/{}/logs_{}_{}.csv'.format(args.model_n, args.dropout, args.dataset_p.replace('/', '-'))
    pd.DataFrame([headers]).to_csv(f_n, mode='w', header=False, index=False)
    run_uq(model_name=args.model_n, checkout_path=models[args.model_n], T=30, dataset=args.dataset_p,
           save_path=f_n, uq_logs=logs, dropout=args.dropout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_n",
                        required=True,
                        default='ssdlite320_mobilenet_v3_large',
                        help="Model name",
                        )
    parser.add_argument("--dataset_p",
                        required=True,
                        default='origimg/test',
                        help="Path of the dataset")
    parser.add_argument("--dropout",
                        required=True,
                        type=float,
                        default=0.1,
                        help="Dropout rate",
                        )

    arg = parser.parse_args()

    main(arg)

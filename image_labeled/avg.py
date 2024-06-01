#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/05/2024 15:19
# @Author  : Chengjie
# @File    : avg.py
# @Software: PyCharm

import json
import numpy as np


def calcu_avg_uq_metrics(f_n):
    f = open(f_n)
    output = json.load(f)

    entropy, mi, tv_cp, tv_box, ps = [], [], [], [], []
    for key in output.keys():
        entropy.append(output[key]['entropy [classification]'])
        mi.append(output[key]['mutual_info [classification]'])
        tv_cp.append(output[key]['total_var_center_point [regression]'])
        tv_box.append(output[key]['total_var_box [regression]'])
        ps.append(output[key]['predictive_surface [regression]'])

    # print(np.array(entropy).mean())

    output.update({'Metrics_Avg': {'entropy [classification]': np.array(entropy).mean(),
                                   'mutual_info [classification]': np.array(mi).mean(),
                                   'total_var_center_point [regression]': np.array(tv_cp).mean(),
                                   'total_var_box [regression]': np.array(tv_box).mean(),
                                   'predictive_surface [regression]': np.array(ps).mean(),
                                   },

                   })

    # print(json.dumps(output, indent=4))
    with open(f_n, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    calcu_avg_uq_metrics(f_n='output_noisy_image_open43.json')

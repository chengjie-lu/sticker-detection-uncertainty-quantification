#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/03/2024 21:45
# @Author  : Chengjie
# @File    : metrics.py
# @Software: PyCharm

import numpy as np


class UQMetrics:
    def __init__(self):
        self.shannon_entropy = 0
        self.mutual_information = 0
        # pass

    def entropy(self, events, ets=1e-15, base=2):
        """
        Shannon Entropy Calculation
        :param events:
        :param ets:
        :param base:
        :return:
        """
        self.shannon_entropy = round(-sum([p * (np.log(p + ets) / np.log(base)) for p in events]), 5)


if __name__ == '__main__':
    uqmetrics = UQMetrics()
    uqmetrics.entropy([0, 1])
    print(uqmetrics.shannon_entropy)

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

    def calcu_entropy(self, events, ets=1e-15, base=2):
        """
        Shannon Entropy Calculation
        :param events:
        :param ets:
        :param base:
        :return:
        """
        self.shannon_entropy = round(-sum([p * (np.log(p + ets) / np.log(base)) for p in events]), 5)
        return self.shannon_entropy

    def calcu_mi(self, events, ets=1e-15, base=2):
        def calcu(e):
            t = 0
            for s in e:
                t += -sum([p * (np.log(p + ets) / np.log(base)) for p in s])
            return t / len(e)

        self.mutual_information = self.calcu_entropy(events=np.mean(np.transpose(events), axis=1)) + calcu(events)
        return self.mutual_information

    def calcu_mutual_information(self, X, Y, Z):
        """
        Calculate mutual information between three discrete random variables X, Y, and Z.
        http://www.scholarpedia.org/article/Mutual_information#:~:text=Mutual%20information%20is%20one%20of,variable%20given%20knowledge%20of%20another.
        Parameters:
            X, Y, Z : array-like, shape (n_samples,)
                Arrays containing discrete random variables.

        Returns:
            mutual_info : float
                The mutual information between X, Y, and Z.
        """
        # Compute joint probability distribution
        unique_X = np.unique(X)
        unique_Y = np.unique(Y)
        unique_Z = np.unique(Z)

        joint_probs = np.zeros((len(unique_X), len(unique_Y), len(unique_Z)))
        for i, x in enumerate(unique_X):
            for j, y in enumerate(unique_Y):
                for k, z in enumerate(unique_Z):
                    joint_probs[i, j, k] = np.sum(np.logical_and(np.logical_and(X == x, Y == y), Z == z)) / float(
                        len(X))

        # Compute marginal probability distributions
        px = np.sum(joint_probs, axis=(1, 2))
        py = np.sum(joint_probs, axis=(0, 2))
        pz = np.sum(joint_probs, axis=(0, 1))

        # Compute mutual information
        mutual_info = 0.0
        for i, x in enumerate(unique_X):
            for j, y in enumerate(unique_Y):
                for k, z in enumerate(unique_Z):
                    if joint_probs[i, j, k] > 0.0:
                        mutual_info += joint_probs[i, j, k] * np.log2(joint_probs[i, j, k] / (px[i] * py[j] * pz[k]))

        self.mutual_information = mutual_info
        return self.mutual_information

    # Example usage:
    # X, Y, and Z are arrays containing discrete random variables
    # mutual_info_score = mutual_information(X, Y, Z)


if __name__ == '__main__':
    uqmetrics = UQMetrics()
    uqmetrics.calcu_entropy([0, 1])
    print(uqmetrics.shannon_entropy)

#!/usr/bin/env python3
"""
L2 regularization cost
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    L2 regularization cost

    :param cost: cost without regularization

    :param lambtha: regularization coefficient

    :param weights: dictionary with weights and biases

    :param L: Number of layers in the network

    :param m: Number of data points

    :return: L2 regularization cost
    """
    reg = 0

    for i in range(1, L + 1):
        weights_i = weights['W' + str(i)]
        reg += np.sum(np.square(weights_i))

    cost_l2 = cost + (lambtha / (2 * m)) * reg

    return cost_l2

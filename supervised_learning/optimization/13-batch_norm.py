#!/usr/bin/env python3
"""
function which normalizes an unactivated output using batch normalization
"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    function which normalizes an unactivated output using batch normalization

    :param Z: unactivated output to be normalized
    m is the number of data points
    n is the number of features in Z

    :param gamma: numoy array (1, n) containing the scales used for
    normalization

    :param beta: numoy array (1, n) containing the offsets used
    for normalization

    :param epsilon: small number to avoid division by zero

    :return: normalized unactivated output
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    std_dev = np.sqrt(var + epsilon)

    Z_norm = (Z - mean) / std_dev

    scaled_Z = Z_norm * gamma + beta

    return scaled_Z

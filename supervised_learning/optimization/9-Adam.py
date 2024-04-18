#!/usr/bin/env python3
"""
Updates a variable using Adam optimization algorithm
"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using Adam optimization algorithm

    :param alpha: learning rate

    :param beta1: weight used for first movement

    :param beta2: weight used for second movement

    :param epsilon: avoid division by zero

    :param var: variable to be updated

    :param grad: gradient of var

    :param v: previous moment of var

    :param s: previous second moment of var

    :param t: time step for bias correction

    :return: updated variable, first moment and second moment of var
    """
    new_v = beta1 * v + (1 - beta1) * grad
    new_s = beta2 * s + (1 - beta2) * grad ** 2

    v_fixed = new_v / (1 - beta1 ** t)
    s_fixed = new_s / (1 - beta2 ** t)

    updated = var - (alpha * (v_fixed / (np.sqrt(s_fixed) + epsilon)))

    return updated, v_fixed, s_fixed

#!/usr/bin/env python3
"""
Updates a variable using a gradient descent algorithm with momentum
optimization.
"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using a gradient descent algorithm with momentum
    optimization.

    :param alpha: learning rate

    :param beta1: momentum weight

    :param var: variable to update

    :param grad: gradient of var

    :param v: previous first movement

    :return: updated variable and new movement
    """
    dW = beta1 * v + (1 - beta1) * grad

    var_new = var - dW * alpha

    return var_new, grad

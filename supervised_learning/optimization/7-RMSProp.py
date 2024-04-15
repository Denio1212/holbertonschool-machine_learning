#!/usr/bin/env python3
"""
Creates the RMSProp training operation for neural network
"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates the RMSProp weights using gradient descent

    :param alpha: learning rate

    :param beta2: RMSProp weight

    :param epsilon: avoid division by zero

    :param var: variable to update

    :param grad: gradient of var

    :param s: previous second moment of var

    :return: updated var and the new moment
    """
    square_grad = beta2 * s + (1 - beta2) * (grad**2)
    updated_var = var - ((alpha * grad) / (np.sqrt(square_grad) + epsilon))

    return updated_var, square_grad

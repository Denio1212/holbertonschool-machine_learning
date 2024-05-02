#!/usr/bin/env python3
"""
Calculates the L2 regularization cost of a neural network.
"""


import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Calculates the L2 regularization cost of a neural network.

    :param cost: Cost function without regularization.

    :return: L2 regularization cost in a tensor parameter
    """
    l2_reg = tf.losses.get_regularization_losses()

    l2_reg_costs = cost + l2_reg

    return l2_reg_costs

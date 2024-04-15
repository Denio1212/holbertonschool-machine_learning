#!/usr/bin/env python3
"""
Updates a variable using a gradient descent algorithm with momentum
optimization.
"""


import numpy as np
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow 1

    :param loss: loss of the network

    :param alpha: learning rate

    :param beta1: momentum weight
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op

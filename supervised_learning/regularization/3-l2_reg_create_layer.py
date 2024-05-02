#!/usr/bin/env python3
"""
Creates a layer in tensorflow 1 that includes a
l2 regularization parameter.
"""


import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a layer in tensorflow 1 that includes l2 regularization

    :param prev: previous layer

    :param n: layer size

    :param activation: activation function

    :param lambtha: regularization parameter

    :return: new layer output
    """
    l2_layer = tf.layers.dense(n, activation=activation,
                               kernel_regularizer=lambtha)(prev)

    return l2_layer

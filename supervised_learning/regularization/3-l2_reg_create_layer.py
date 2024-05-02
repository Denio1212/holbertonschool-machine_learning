#!/usr/bin/env python3
"""
Creates a layer in tensorflow 1 that includes a
l2 regularization parameter.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a layer in tensorflow 1 that includes l2 regularization

    :param prev: previous layer

    :param n: layer size

    :param activation: activation function

    :param lambtha: regularization parameter

    :return: new layer output
    """
    regualizer = tf.keras.regularizers.l2(lambtha)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    l2_layer = tf.layers.dense(n, activation=activation,
                               kernel_initializer=init,
                               kernel_regularizer=regualizer,
                               name="l2_layer")
    output = l2_layer(prev)

    return output

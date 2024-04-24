#!/usr/bin/env python3
"""
Batch Normalization in tensorflow 1.x
"""

import numpy as np
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create batch normalization layer for neural network

    :param prev: previous layer

    :param n: number of nodes to be created

    :param activation: activation function

    beta and gama are two trainable params

    :return: batch normalization layer
    """
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    new_layer =tf.keras.layers.Dense(n,
                                     activation=None,
                                     kernel_initializer=init,
                                     name="layer")

    x = new_layer(prev)
    mean, variance = tf.nn.moments(x, axes=[0])

    beta = tf.Variable(tf.zeros([n]), name="beta")
    gamma = tf.Variable(tf.ones([n]), name="gamma")

    epsilon = 1e-8

    x_norm = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)

    return activation(x_norm)

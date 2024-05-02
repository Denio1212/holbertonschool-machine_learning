#!/usr/bin/env python3
"""
Creates a layer of a neural network with a dropout layer.
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network with a dropout layer.

    :param prev: The previous layer.

    :param n: The number of nodes

    :param activation: The activation function

    :param keep_prob: The dropout probability to keep the weights.

    :return: The created layer.
    """
    dropout = tf.keras.layers.Dropout(rate=keep_prob)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    dropout_layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=dropout,
        name="dropout_layer"
    )
    output = dropout_layer(prev)

    return output

#!/usr/bin/env python3
"""
Creates the training program for a neural network in tensorflow 1
with the adam optimizer
"""


import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training program for a neural network in tensorflow 1
    with the adam optimizer

    :param loss: the loss function

    :param alpha: the learning rate

    :param beta1:  weight used for the first moment

    :param beta2:  weight used for the second moment

    :param epsilon: avoid division by zero

    :return: the training program for the adam optimizer
    """
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                  beta2=beta2, epsilon=epsilon).minimize(loss)

#!/usr/bin/env python3
"""
Creates the RMSProp training operation for neural network using tensorflow 1
"""


import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
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
    optimiser = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    train_op = optimiser.minimize(loss)

    return train_op

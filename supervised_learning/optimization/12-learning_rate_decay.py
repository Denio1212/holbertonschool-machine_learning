#!/usr/bin/env python3
"""
Creates a learning rate decay function for inverse time decay in tensorflow 1
"""

import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay function for inverse time decay
    in tensorflow 1

    :param alpha: original learning rate

    :param decay_rate: decay rate of alpha

    :param global_step: number of iterations that have passed

    :param decay_step: number of iterations that have passed in decay_rate

    :return: learning rate decay
    """
    learning_rate = tf.train.inverse_time_decay(learning_rate=alpha,
                                               decay_rate=decay_rate,
                                               global_step=global_step,
                                               decay_steps=decay_step,
                                               staircase=True)

    return learning_rate

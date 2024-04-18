#!/usr/bin/env python3
"""
Updates Learning rate decay using inverse time decay in numpy
"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates Learning rate decay using inverse time decay in numpy

    :param alpha: learning rate

    :param decay_rate: rate of alpha decay

    :param global_step: current step

    :param decay_step: decay step

    The learning rate should occur in a stepwise fashion

    :return: updated value of learning rate (alpha)
    """
    bottom_shelf = 1 + decay_rate * (global_step // decay_step)

    alpha = alpha / bottom_shelf

    return alpha

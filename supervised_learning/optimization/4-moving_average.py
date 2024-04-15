#!/usr/bin/env python3
"""
Calculates the weighted moving average of a data set.
"""


import numpy as np


def moving_average(data, beta):
    """
    data is the list of data to calculate the moving average of

    beta is the weight used for the moving average

    Your moving average calculation should use bias correction

    Returns: a list containing the moving averages of data
    """
    moving_averages = []
    w = 0

    for i, d in enumerate(data):
        w = beta * w + (1 - beta) * d
        w_new = w / (1 - beta**(i+1))
        moving_averages.append(w_new)

    return moving_averages

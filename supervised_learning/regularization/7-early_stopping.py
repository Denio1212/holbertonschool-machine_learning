#!/usr/bin/env python3
"""
Determines whether a model should be stopped early.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether a model should be stopped early.

    :param cost: Validation Cost

    :param opt_cost: Validation Opt Cost optimal

    :param threshold: Validation Threshold for early stopping

    :param patience: Patience for early stopping

    :param count: Number of iterations that have been met
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return count == patience, count

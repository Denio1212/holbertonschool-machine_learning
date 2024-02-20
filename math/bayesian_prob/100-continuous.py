#!/usr/bin/env python3
"""
Based on 3-posterior this function calculates the continuous posterior
probability
"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    calculate the continuous posterior function
    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param p1: is the lower bound on the range
    :param p2: is the upper bound on the range
    :return: the posterior probability that p is within the range
    [p1, p2] given x and n
    """
    return special.betainc(x + 1, n - x + 1, p1) - special.betainc(x + 1, n + x - 1, p2)

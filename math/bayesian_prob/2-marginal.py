#!/usr/bin/env python3
"""
Based on the previous exercise this function calculates the marginal
probability of calculating the data.
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data
    :param x: number of patients that develop side effects
    :param n: total number of patients observed
    :param P: 1D numpy array with the various hypothetical probabilities
    :param Pr: 1D numpy array with the prior beliefs about P
    :return: marginal probability of obtaining x and n
    Im too lazy to write the conditions so yea!
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or"
                         " equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same"
                        " shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    n_x_fact = np.math.factorial(n - x)

    factorial_part = n_fact / (x_fact * n_x_fact)

    intersection_results = (factorial_part * (P ** x) * ((1 - P) ** (n - x))
                            * Pr)
    marginal_prob = np.sum(intersection_results)
    return marginal_prob

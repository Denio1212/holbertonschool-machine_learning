#!/usr/bin/env python3
"""
Considering the previous function we will calculate the posterior probability
of the various hypothetical probabilities
"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    :param x: the nummber of patients with the side effect
    :param n: total number of patients observed
    :param P: 1D numpy array containing the various hypothetical probabilities
    :param Pr: 1D numpy array containing the prior probabilities of P
    :return: Posterior probability of each probability P given x and n
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
    marginal_results = np.sum(intersection_results)

    posterior_results = intersection_results / marginal_results
    return posterior_results

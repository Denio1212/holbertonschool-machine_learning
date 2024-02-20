#!/usr/bin/env python3
"""
Calculates the intersection based on the 0-likelihood file previous.
It calculates the intersection of obtaining this data with the various
hypothetical probabilities
"""
import numpy as np
from scipy import special


def intersection(x, n, P, Pr):
    """
    Calculates the intersection from the data given.
    :param x: number of patients with side effects
    :param n: total number of patients
    :param P: 1D numpy array containing various probabilities of side effects
    :param Pr: 1D numpy array containing the prior probabilities of side effects
    :return: 1D numpy array containing the intersection of obtaining x and n
    with each probability in P, respectively.

    if n is not a positive integer raise:
    ValueError: "n must be a positive integer"

    if x is not an integer that is greater than or equal to 0 raise:
    ValueError: "x must be an integer that is greater than or equal to 0"
    if x is greater than n raise:
    ValueError: "x cannot be greater than n"

    if P is not a 1D numpy array raise:
    TypeError: "P must be a 1D numpy.ndarray"

    If Pr is not a numpy array with the same shape as P raise:
    TypeError: "Pr must be a numpy.ndarray with the same shape as P"

    If any value of P, Pr is not in range[0, 1] raise:
    ValueError: "All values in {P} must be in the range [0, 1]"

    if Pr does not sum to 1 raise:
    ValueError: "Pr must sum to 1"
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1) or np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in P and Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    scipy_special_binom = (np.math.factorial(n) // (np.math.factorial(x) *
                           np.math.factorial(n - x)))
    likelihood = scipy_special_binom * np.power(P, x) * np.power(1.0 - P, n - x)
    intersections = likelihood * Pr
    return intersections

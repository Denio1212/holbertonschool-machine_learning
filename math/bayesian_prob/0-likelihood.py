#!/usr/bin/env python3
"""
You are conducting a study on a revolutionary cancer drug and are looking
to find the probability that a patient who takes this drug will develop severe
side effects.
During your trials, n patients take the drug and x patients develop severe side
 effects.  You can assume that x follows a binomial distribution.

 The function calculates the likelihood of obtaining this data given various
 hypothetical probabilities of developing side effects.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculate the hypothetical likelihood of severe side effects
    :param x: number of patients that develop side effects
    :param n: total number of patients
    :param P: numpy array of probabilities of developing side effects
    :return: 1d numpy array of the likelihood of obtaining the data x adn n
    for each probability P respectively

    if n is not positive int raise:
    ValueError: n must be a positive integer

    if x is not an int that is greater than or equal to 0 raise:
    Value Error: x must be an integer that is greater than or equal to 0
    if x is greater than n raise:
    ValueError: x must be greater than n

    if p is not a 1d numpy array raise:
    Type error: p must be a 1d numpy.ndarray
    if any value in p is not in the range [0, 1] raise:
    ValueError: All values in P must be in the range [0, 1]
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than "
                         "or equal to 0")
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1d numpy.ndarray")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    n_factorial = np.math.factorial(n)
    x_factorial = np.math.factorial(x)
    likelihoods = ((n_factorial / (x_factorial * np.math.factorial(n - x)))
                   * (P ** x) * ((1 - P) ** (n - x)))
    return likelihoods

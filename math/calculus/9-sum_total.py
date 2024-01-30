#!/usr/bin/env python3
"""
Calculates the sigma notation of a given n
"""


def summation_i_squared(n):
    """
    Summarizes the sigma of given n
    Args:
        n: The Iterator

    Returns: The sum
    """
    return sum(i ** 2 for i in range(1, n + 1))

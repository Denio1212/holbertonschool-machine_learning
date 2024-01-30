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
    return int(n * (n + 1) * (2 * n + 1) / 6)

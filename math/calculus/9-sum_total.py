#!/usr/bin/env python3
"""
Calculates the sigma notation of a given n
"""


def summation_i_squared(n):
    """
    Summarizes the sigma of given n without loops
    Args:
        n: The Iterator

    Returns: The sum
    """
    if n is type(str):
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)

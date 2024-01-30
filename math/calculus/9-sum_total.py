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
    if isinstance(n, int) and n > 0:
        return int(n * (n + 1) * (2 * n + 1) / 6)
    else:
        return None

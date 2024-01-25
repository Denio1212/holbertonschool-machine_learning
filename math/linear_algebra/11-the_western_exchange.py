#!/usr/bin/env python3
"""
Returns a transposed matrix of a given numpy array
"""


def np_transpose(matrix):
    """
    Transposes a numpy array
    Args:
        matrix: The numpy array

    Returns: A new numpy array
    """
    trans = matrix.array([])
    trans = matrix.transpose(matrix)
    return trans

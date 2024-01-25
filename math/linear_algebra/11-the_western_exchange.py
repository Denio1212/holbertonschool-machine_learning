#!/usr/bin/env python3
import numpy as np
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
    trans = np.array([])
    trans = np.transpose(matrix)
    return trans

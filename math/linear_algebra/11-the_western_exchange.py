#!/usr/bin/env python3
"""
Returns a transposed matrix of a given numpy array
"""
import numpy as np


def np_transpose(matrix):
    """
    Transposes a numpy array
    Args:
        matrix: The numpy array

    Returns: A new numpy array
    """
    trans = []
    trans = np.transpose(matrix)
    return trans

#!/usr/bin/env python3
"""
Concatenate two matrices along an axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two matrices along an axis
    Args:
        mat1: first matrix to be concatenated
        mat2: second matrix to be concatenated
        axis: the given axis

    Returns: A concatenated np.ndarray
    """
    return np.concatenate((mat1, mat2), axis=axis)

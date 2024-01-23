#!/usr/bin/env python3
"""
Adds two arrays element-wise but on a 2d matrix
"""


def add_matrices2D(arr1, arr2):
    """
    Args:
        arr1: First array with 2d matrix
        arr2: Second array with 2d matrix

    Returns:
    A new array with the sum of each element from the same index
    from each of the arrays
    """
    added_arrays = []
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        return None
    for row1, row2 in zip(arr1, arr2):
        for i, j in zip(row1, row2):
            added_arrays.append(i + j)
    return added_arrays[:2], added_arrays[2:]

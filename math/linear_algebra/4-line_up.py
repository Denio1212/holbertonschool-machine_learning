#!/usr/bin/env python3
"""
Adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Args:
        arr1: First array
        arr2: Second array

    Returns:
    A new array with the sum of each element from the same index
    from each of the arrays
    """
    added_arrays = []
    if len(arr1) != len(arr2):
        return None
    for i, j in zip(arr1, arr2):
        added_arrays.append(i + j)
    return added_arrays

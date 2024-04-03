#!/usr/bin/env python3
"""
Decodes the output of the one hot encoding
"""


import numpy as np


def one_hot_decode(one_hot):
    """
    Decodes the output of the one hot encoding

    :param one_hot: output of the one hot encoding
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    decoded = np.argmax(one_hot, axis=0)
    return decoded

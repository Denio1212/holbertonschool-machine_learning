#!/usr/bin/env python3
"""
Function which performs forward propagation on the input data over
a pooling neural network layer
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function which performs forward propagation on the input data over
    a pooling neural network layer

    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m is number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels of the previous layer

    :param kernel_shape: tuple of length 2 (kh, hw), containing the strides
    of the pooling operation
    sh is the stride of height
    sw is the stride of width

    :param stride: tuple of length 2 (sh, sw), containing the strides

    :param mode: 'max' or 'avg'

    :return: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_height = int((h_prev - kh) / sh + 1)
    out_width = int((w_prev - kw) / sw + 1)

    Pooled = np.zeros((m, out_height, out_width, c_prev))

    for i in range(out_height):
        for j in range(out_width):
            image_zone = A_prev[:, i * sh:i * sh + kh,
                                j * sw:j * sw + kw, :]

            if mode == 'max':
                Pooled[:, i, j, :] = np.max(image_zone, axis=(1, 2))
            elif mode == 'avg':
                Pooled[:, i, j, :] = np.average(image_zone, axis=(1, 2))

    return Pooled

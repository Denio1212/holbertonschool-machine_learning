#!/usr/bin/env python3
"""
Function which performs forward propagation on the input data over
a convolutional neural network layer
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function which performs forward propagation on the input data

    :param A_prev: array of input data with shape (m, h_prev, w_prev, c_prev)
    m is number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels of the previous layer

    :param W: array of kernels w shape (kh, kw, c_prev, c_new)
    kh is the filter height
    kw is the filter width
    c_prev is the number of channels in the previous layer
    c_new is the number of channels of the output

    :param activation: activation function

    :param padding: padding mode, a string 'same' or 'valid', indicates
    padding on both sides

    :param stride: stride mode, a tuple (sh, sw)
    sh is the stride height
    sw is the stride width
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "valid":
        ph = 0
        pw = 0
    elif padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)

    out_height = int((h_prev - kh + 2 * ph) / sh) + 1
    out_width = int((w_prev - kw + 2 * pw) / sw) + 1

    Z = np.zeros((m, out_height, out_width, c_new))

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), 'constant')

    for k in range(c_new):
        for h in range(out_height):
            for w in range(out_width):
                # extract region from each image
                image_zone = A_prev_padded[:, h * sh:h * sh + kh,
                                           w * sw:w * sw + kw, :]

                # element wize multiplication
                Z[:, h, w, k] = np.sum(image_zone
                                       * W[:, :, :, k],
                                       axis=(1, 2, 3))

    Z = Z + b

    Z = activation(Z)

    return Z

#!/usr/bin/env python3
"""
L2 Regularization with Gradient Descent
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    L2 Regularization with Gradient Descent

    """
    m = Y.shape[1]
    grad = cache["A" + str(L)] - Y

    for layer in range(L, 0, -1):
        L2 = lambtha / m * weights["W" + str(layer)]

        A_prev = cache["A" + str(layer - 1)]

        dW = np.matmul(grad, A_prev.T) / m + L2
        db = np.sum(grad, axis=1, keepdims=True) / m
        dA = np.matmul(weights["W" + str(layer)].t, grad)

        if layer != 1:
            grad = dA * (1 - np.power(A_prev, 2))
        else:
            grad = dA

        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db

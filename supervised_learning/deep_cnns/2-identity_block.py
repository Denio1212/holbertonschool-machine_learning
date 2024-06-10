#!/usr/bin/env python3
"""
Makes the ResNet model identity block.
"""

from tensorflow import keras


def identity_block(A_prev, filters):
    """
    Identity block function.

    :param A_prev: output of the previous layer.

    :param filters: number of filters in tuple form carrying:
    F11 - filter for 1x1 convolution
    F3 - filter size for 3x3 convolution
    F22 - filter for second 1z1 convolution

    All convolutions will be followed by Batch normalization, and ReLU activation.

    All weights will use normal initialization.

    Returns: Activated output of the block.
    """
    F11, F3, F12 = filters
    initializer = keras.initializers.he_normal(seed=0)
    activation = keras.activations.relu

    layers = keras.layers

    Conv_1x1 = layers.Conv2D(
        F11,
        (1, 1),
        padding='same',
        kernel_initializer=initializer,
    )(A_prev)
    Batch_1x1 = layers.BatchNormalization(axis=3)(Conv_1x1)
    ReLU_1x1 = layers.Activation(activation)(Batch_1x1)

    Conv_3x3 = layers.Conv2D(
        F3,
        (3, 3),
        padding='same',
        kernel_initializer=initializer,
    )(ReLU_1x1)
    Batch_3x3 = layers.BatchNormalization(axis=3)(Conv_3x3)
    ReLU_3x3 = layers.Activation(activation)(Batch_3x3)

    Conv_1x1_2 = layers.Conv2D(
        F12,
        (1, 1),
        padding='same',
        kernel_initializer=initializer,
    )(ReLU_3x3)
    Batch_1x1_2 = layers.BatchNormalization(axis=3)(Conv_1x1_2)

    pre_output = layers.Add()([Batch_1x1_2, A_prev])

    output = layers.Activation(activation)(pre_output)

    return output

#!/usr/bin/env python3
"""
Makes the Projection blocks of the ResNet Architecture
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Projection block function.

    :param A_prev: output of the previous layer.

    :param filters: number of filters in tuple form carrying:
    F11 - filter for 1x1 convolution
    F3 - filter size for 3x3 convolution
    F12 -filter for second 1z1 convolution as well as the shortcut convolution

    s- stride of the main and shortcut convolution

    All convolutions will be followed by Batch normalization,
    and ReLU activation.

    All weights will use he_normal initialization.
    Seed of he_normal initialization will be set to 0

    Returns: Activated output of the block.
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)
    activation = K.activations.relu
    layers = K.layers
    batch = layers.BatchNormalization

    Conv_1x1 = layers.Conv2D(
        F11,
        (1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer,
    )(A_prev)
    Batch_1x1 = batch(axis=3)(Conv_1x1)
    ReLU_1x1 = layers.Activation(activation)(Batch_1x1)

    Conv_3x3 = layers.Conv2D(
        F3,
        (3, 3),
        padding='same',
        kernel_initializer=initializer,
    )(ReLU_1x1)
    Batch_3x3 = batch(axis=3)(Conv_3x3)
    ReLU_3x3 = layers.Activation(activation)(Batch_3x3)

    Conv_1x1_2 = layers.Conv2D(
        F12,
        (1, 1),
        padding='same',
        kernel_initializer=initializer,
    )(ReLU_3x3)
    Batch_1x1_2 = batch(axis=3)(Conv_1x1_2)

    skip_layer = layers.Conv2D(
        F12,
        (1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer,
    )(A_prev)
    skip_Batch = batch(axis=3)(skip_layer)

    pre_output = layers.Add()([Batch_1x1_2, skip_Batch])

    output = layers.Activation(activation)(pre_output)

    return output

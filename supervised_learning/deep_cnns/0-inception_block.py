#!/usr/bin/env python3
"""
makes the Google inception block
"""


import tensorflow.keras as keras


def inception_block(A_prev, filters):
    """
    Makes the Google inception block

    :param A_prev: input tensor from previous layer

    :param filters: filters is a tuple or list containing
    F1, F3R, F3,F5R, F5, FPP, respectively

    all conv layers must use ReLU activation function

    Returns: the concatenated output of the inception block
"""
    layers = keras.layers
    f1 = filters[0]
    f3r = filters[1]
    f3 = filters[2]
    f5r = filters[3]
    f5 = filters[4]
    fpp = filters[5]

    tower_1 = layers.Conv2D(
        filters=f1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
    )

    tower_2 = layers.Conv2D(
        filters=f3r,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
    )

    tower_3 = layers.Conv2D(
        filters=f3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
    )

    tower_4 = layers.Conv2D(
        filters=f5r,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
    )

    tower_5 = layers.Conv2D(
        filters=f5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
    )

    pool = layers.MaxPooling2D((3, 3), strides=(1, 1),
                               padding='same')(A_prev)

    tower_6 = layers.Conv2D(
        filters=fpp,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
    )

    layer_1 = tower_1(A_prev)
    layer_3_prev = tower_2(A_prev)
    layer_3 = tower_3(layer_3_prev)
    layer_5_prev = tower_4(A_prev)
    layer_5 = tower_5(layer_5_prev)
    layer_end = tower_6(pool)

    output = layers.concatenate([layer_1, layer_3, layer_5, layer_end])

    return output

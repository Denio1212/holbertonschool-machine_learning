#!/usr/bin/env python3
"""
makes the Google inception architecture
using the inception block made previously
"""


import tensorflow.keras as keras
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Makes the Google Inception Network

    The input shape is (224, 224, 3)

    Returns: The keras model
    """
    dense = keras.layers.Dense
    conv = keras.layers.Conv2D
    max_pool = keras.layers.MaxPooling2D
    inception = inception_block
    avg_pool = keras.layers.AveragePooling2D
    drop = keras.layers.Dropout

    input_layer = keras.layers.Input(shape=(224, 224, 3))

    x = conv(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        strides=(2, 2),
        activation='relu',
        name='conv_start',
    )(input_layer)

    x = max_pool(
        (3, 3),
        strides=(2, 2),
        padding='same',
        name='max_pool_1',
    )(x)

    x = conv(
        filters=64,
        kernel_size=(1, 1),
        padding='same',
        strides=(1, 1),
        activation='relu',
        name="conv_pre_3x3"
    )(x)

    x = conv(
        filters=192,
        kernel_size=(3, 3),
        padding='same',
        strides=(1, 1),
        activation='relu',
        name="conv_3x3"
    )(x)

    x = max_pool(
        (3, 3),
        strides=(2, 2),
        padding='same',
        name='max_pool_2',
    )(x)

    x = inception_block(
        x,
        [64, 96, 128, 16, 32, 32],
    )

    x = inception_block(
        x,
        [128, 128, 192, 32, 96, 64],
    )

    x = max_pool(
        (3, 3),
        strides=(2, 2),
        padding='same',
        name='max_pool_inception_1',
    )(x)

    x = inception_block(
        x,
        [192, 96, 208, 16, 48, 64],
    )

    x1 = avg_pool(
        (5, 5),
        strides=3
    )(x)

    x1 = conv(
        filters=128,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
    )(x1)

    x1 = keras.layers.Flatten()(x1)

    x1 = dense(
        1024,
        activation='relu',
    )(x1)

    x1 = drop(0.7)(x1)

    x1 = dense(10, activation='softmax', name="aux_output_1")(x1)

    x = inception_block(
        x,
        [160, 112, 224, 24, 64, 64],
    )

    x = inception_block(
        x,
        [128, 128, 256, 24, 64, 64],
    )

    x = inception_block(
        x,
        [112, 144, 288, 32, 64, 64],
    )

    x2 = avg_pool(
        (5, 5),
        strides=3
    )(x)

    x2 = conv(
        filters=128,
        kernel_size=(1, 1),
        padding="same",
        activation="relu"
    )(x2)

    x2 = keras.layers.Flatten()(x2)

    x2 = dense(
        1024,
        activation="relu"
    )(x2)

    x2 = drop(0.7)(x2)

    x2 = dense(10, activation='softmax', name="aux_output_2")(x2)

    x = inception_block(
        x,
        [256, 160, 320, 32, 128, 128],
    )

    x = max_pool(
        (3, 3),
        strides=(2, 2),
        padding='same',
        name='max_pool_inception_2',
    )(x)

    x = inception_block(
        x,
        [384, 192, 384, 48, 128, 128],
    )

    x = inception_block(
        x,
        [384, 192, 384, 48, 128, 128],
    )

    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = drop(0, 0.4)(x)

    x = dense(10, activation='softmax', name="main_output")(x)

    keras_model = keras.models.Model(inputs=input_layer, outputs=x)

    return keras_model

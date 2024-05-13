#!/usr/bin/env python3
"""
    LeNet-5 (Keras implementation)
"""

import tensorflow.keras as keras


def lenet5(X):
    """
    LeNet-5 (Keras implementation)

    :param X: input data shaped [m, 28, 28, 1] containing 28x28x1 input images
    m -> number of input images
    """
    initializer = keras.initializers.HeNormal()
    layers = keras.layers

    model = keras.Sequential([
        layers.Conv2D(6,
                      kernel_size=5,
                      padding='same',
                      activation='relu',
                      kernel_initializer=initializer),

        layers.MaxPooling2D(pool_size=2, strides=2),

        layers.Conv2D(
            16,
            kernel_size=5,
            padding='valid',
            activation='relu',
            kernel_initializer=initializer
        ),

        layers.MaxPooling2D(pool_size=2,
                            strides=2),

        layers.Flatten(),

        layers.Dense(120, activation='relu', kernel_initializer=initializer),

        layers.Dense(84, activation='relu', kernel_initializer=initializer),

        layers.Dense(10, activation='softmax', kernel_initializer=initializer)
    ])

    model = model.compile(loss='sparce_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy']
                          )

    return model

#!/usr/bin/env python3
"""
Makes the ResNet Architecture
"""

from tensorflow import keras
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    ResNet50 architecture.

    The input image is 224x224 RGB.

    All convolutions will be followed by Batch normalization,
    and ReLU activation.

    All weights will use he_normal initialization.
    Seed of he_normal initialization will be set to 0

    Returns: Keras Model
    """
    init = keras.initializers.he_normal()
    activation = keras.activations.relu
    input = keras.Input(shape=(224, 224, 3))
    layers = keras.layers

    start = layers.Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init,
    )(input)
    start_batch = layers.BatchNormalization(axis=3)(start)
    start_relu = layers.Activation(activation)(start_batch)

    max_pool_1 = (layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same',
    )(start_relu))

    project_1 = projection_block(start_relu, [64, 64, 256], s=1)
    identity_1 = identity_block(
        project_1,
        [64, 64, 256],
    )
    identity_2 = identity_block(
        identity_1,
        [64, 64, 256],
    )

    project_2 = projection_block(identity_2, [128, 128, 512], s=2)
    identity_2_1 = identity_block(
        project_2,
        [128, 128, 512]
    )
    identity_2_2 = identity_block(
        identity_2_1,
        [128, 128, 512],
    )
    identity_2_3 = identity_block(
        identity_2_2,
        [128, 128, 512],
    )

    project_3 = projection_block(identity_2_3, [256, 256, 1024], s=2)
    identity_3_1 = identity_block(
        project_3,
        [256, 256, 1024],
    )
    identity_3_2 = identity_block(
        identity_3_1,
        [256, 256, 1024],
    )
    identity_3_3 = identity_block(
        identity_3_2,
        [256, 256, 1024],
    )
    identity_3_4 = identity_block(
        identity_3_3,
        [256, 256, 1024],
    )
    identity_3_5 = identity_block(
        identity_3_4,
        [256, 256, 1024],
    )

    project_4 = projection_block(identity_3_5, [512, 512, 2048], s=2)
    identity_4_1 = identity_block(
        project_4,
        [512, 512, 2048],
    )
    identity_4_2 = identity_block(
        identity_4_1,
        [512, 512, 2048],
    )

    average_pooling = layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding="valid"
    )(identity_4_2)

    output = layers.Dense(
        1000,
        activation="softmax",
        kernel_initializer=init,
    )(average_pooling)

    model = keras.models.Model(inputs=input, outputs=output)

    return model

#!/usr/bin/env python3
"""
Writes a function that builds a neural network with the keras library
"""


import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds the neural network model

    :param nx: the number of input features

    :param layers: list containing the number of nodes in the network

    :param activations: list with the activation functions for the layers

    :param lambtha: l2 regularization parameter

    :param keep_prob: keep probability for the dropout layer

    :return: keras model
    """
    model = keras.Sequential()
    for i in range(len(layers)):
        model.add(keras.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=keras.regularizers.l2(lambtha),
            input_dim=nx
        ))
        if i < (len(layers)) - 1 and keep_prob is not None:
            model.add(keras.layers.Dropout(keep_prob))

    return model

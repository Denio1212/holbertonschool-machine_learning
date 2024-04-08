#!/usr/bin/env python3
"""
Adam optimisation using Keras with categorical crossentropy loss function
"""


import tensorflow.keras as keras


def optimize_model(network, alpha, beta1, beta2):
    """
    Optimise the model using categorical crossentropy

    :param network: Keras model to be optimized

    :param alpha: learning rate

    :param beta1: first adam optimiser parameter

    :param beta2: second adam optimiser parameter

    returns None
    """
    adam_optimizer = keras.optimizers.Adam(learning_rate=alpha, beta_1=beta1,
                                           beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=adam_optimizer)
    return None

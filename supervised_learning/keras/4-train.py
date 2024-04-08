#!/usr/bin/env python3
"""
Trains a model using mini batch gradient descent
"""

import tensorflow.keras as keras


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini batch gradient descent

    :param network: model to train

    :param data: numpy array of training data with shape (m, nx)

    :param labels: one hot numpy array of shape (m, classes) with data labels

    :param batch_size: size of batch used for mini batch gradient descent

    :param epochs: number of passes through the data

    :param verbose: boolean that determines whether output should pe printed

    :param shuffle: s a boolean that determines whether to shuffle
    the batches every epoch
    Normally, it is a good idea to shuffle,
    but for reproducibility, we have chosen to set the default to False.

    :return: history of the model
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history

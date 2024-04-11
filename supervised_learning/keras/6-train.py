#!/usr/bin/env python3
"""
Trains a model using mini batch gradient descent
"""

import tensorflow.keras as keras


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
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

    :param early_stopping: boolean that determines when early stopping
    is triggered

    :param patience: the patience for early stopping

    :return: history of the model
    """
    if early_stopping and validation_data is not None:
        callbacks = []
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)]
    else:
        callbacks = None
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
    return history

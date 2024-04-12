#!/usr/bin/env python3
"""
Trains a model using mini batch gradient descent
"""

import tensorflow.keras as keras


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
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

    :param learning_rate_decay: boolean that determines whether to decay
    learning rate formula is the one used in the loss function

    :param alpha: initial decay rate

    :param decay_rate: decay rate

    :return: history of the model
    """
    callbacks = []
    if early_stopping and validation_data:
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)]
    else:
        callbacks = None
    if learning_rate_decay and validation_data:
        def learn_schedule(epochs):
            learning_rate = alpha / (1 + decay_rate * epochs)
            return learning_rate

        callbacks = [keras.callbacks.LearningRateScheduler(learn_schedule,
                                                           verbose=1)]

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

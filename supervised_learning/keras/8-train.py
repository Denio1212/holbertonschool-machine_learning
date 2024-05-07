#!/usr/bin/env python3
"""
Trains a model using mini batch gradient descent
"""

import tensorflow.keras as keras


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
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

    :param alpha: initial decay rate

    :param decay_rate: decay rate

    :param save_best: boolean that determines whether to save the best model

    :param filepath: path to save the model

    :return: history of the model
    """
    callback = []
    if early_stopping is True and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)

        # add to callback list
        callback.append(early_stop)

    if learning_rate_decay and validation_data:
        # function calculate new learning rate
        def scheduler(epochs):
            lr = alpha / (1 + decay_rate * epochs)
            return lr

        inv_time_decay = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1)

        # add to callback list
        callback.append(inv_time_decay)

    # save best model
    if save_best:
        save_best_model = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )

        callback.append(save_best_model)

    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          callbacks=[callback],
                          verbose=verbose,
                          shuffle=shuffle)

    return history

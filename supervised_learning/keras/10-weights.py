#!/usr/bin/env python3
"""
Same as previous tasks but with weights
"""


import tensorflow.keras as keras


def save_weights(network, filename, save_format='h5'):
    """
    Saves weights

    :param network: keras model whose weights will be saved

    :param filename: path to save the weights

    :param save_format: format of the weights

    :return: None
    """
    network.save_weights(filepath=filename,
                         save_format=save_format)

    def load_weights(network, filename):
        """
            function that loads a model's weights

            :param network: model to which the weights should be loaded
            :param filename: path of the file where weights to be loaded

            :return: None
        """
        network.load_weights(filepath=filename)

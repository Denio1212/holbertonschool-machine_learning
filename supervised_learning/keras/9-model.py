#!/usr/bin/env python3
"""
Makes a save model and a load model, which do exactly what they say
"""


from tensorflow import keras


def save_model(network, filename):
    """
    Saves a keras model

    :param network: the model to be saved

    :param filename: the path to save the model

    :return: None
    """
    network.save(filename)

def load_model(filename):
    """
    Loads a keras model

    :param filename: the model to be loaded

    :return: the loaded model
    """
    return keras.models.load_model(filename)

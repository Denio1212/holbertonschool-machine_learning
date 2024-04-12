#!/usr/bin/env python3
"""
Same as previous tasks but with weights
"""


import tensorflow.keras as keras


def save_weights(network, filename, save_format='h5'):
    """
    Saves weights
    """
    network.save_weights(filename=filename, save_format=save_format)


def load_weights(network, filename):
    """
    Load weights
    """
    network.load_weights(filename=filename)

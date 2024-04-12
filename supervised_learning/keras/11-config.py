#!/usr/bin/env python3
"""
    Save and load configuration
"""

import tensorflow.keras as keras


def save_config(network, filename):
    """
    saves a configuration in json format
    """
    network_config = network.to_json(filename)
    with open(filename, 'w') as f:
        f.write(network_config)


def load_config(filename):
    """
    loads a configuration in json format
    """
    with open(filename, 'r') as f:
        lead_network_config = f.read(f)
    return keras.models.model_from_json(lead_network_config)

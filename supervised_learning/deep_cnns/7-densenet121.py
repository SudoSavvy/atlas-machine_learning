#!/usr/bin/env python3
"""Builds the DenseNet-121 architecture"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transitional_layer').transitional_layer


def densenet121(growth_rate=31, compression=1.0):
    """Builds the DenseNet architecture"""

    
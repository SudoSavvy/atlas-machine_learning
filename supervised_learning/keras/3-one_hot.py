#!/usr/bin/env python3
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Parameters:
    - labels: Vector containing label indices
    - classes: Number of classes (optional, inferred if None)

    Returns:
    - One-hot encoded matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)

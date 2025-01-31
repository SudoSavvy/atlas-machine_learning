#!/usr/bin/env python3
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): A one-hot encoded numpy
        array with shape (classes, m).

    Returns:
        numpy.ndarray: A numpy array with shape (m,)
        containing the numeric labels for each example,
        or None on failure.
    """
    # Validate input
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.ndim != 2:
        return None

    # Decode the one-hot matrix
    labels = np.argmax(one_hot, axis=0)

    return labels

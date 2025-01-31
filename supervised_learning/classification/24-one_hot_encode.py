#!/usr/bin/env python3
import numpy as np

def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): A numpy array with shape (m,) containing numeric class labels.
        classes (int): The maximum number of classes found in Y.

    Returns:
        numpy.ndarray: A one-hot encoding of Y with shape (classes, m), or None on failure.
    """
    # Validate input types
    if not isinstance(Y, np.ndarray):
        return None
    if not isinstance(classes, int):
        return None
    if classes < 2:
        return None

    # Validate that classes is larger than the largest element in Y
    if np.max(Y) >= classes:
        return None

    # Initialize the one-hot matrix with zeros
    m = Y.shape[0]
    one_hot = np.zeros((classes, m), dtype=float)

    # Fill the one-hot matrix
    for i, label in enumerate(Y):
        if label < 0:
            return None  # Ensure labels are non-negative
        one_hot[label, i] = 1.0

    return one_hot
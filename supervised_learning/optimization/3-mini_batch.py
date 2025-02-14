#!/usr/bin/env python3
"""
Functions to calculate the normalization constants of a matrix,
normalize it,
shuffle data points in two matrices consistently, and create
mini-batches for training.
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature in a matrix.

    Parameters:
    X (numpy.ndarray): A matrix of shape (m, nx) where m is the number of
                        data points and nx is the number of features.

    Returns:
    tuple: A tuple containing the mean and standard deviation of
    each feature,
           respectively.
    """
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0, ddof=0)
    return mean, std_dev


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
    X (numpy.ndarray): A matrix of shape (d, nx) to normalize
        d is the number of data points
        nx is the number of features
    m (numpy.ndarray): A numpy array of shape (nx,) containing the
    mean of all features of X
    s (numpy.ndarray): A numpy array of shape (nx,) containing the
    standard deviation of all features of X

    Returns:
    numpy.ndarray: The normalized X matrix
    """
    return (X - m) / s


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
    X (numpy.ndarray): The first matrix of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y (numpy.ndarray): The second matrix of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y

    Returns:
    tuple: The shuffled X and Y matrices
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for training using mini-batch gradient descent.

    Parameters:
    X (numpy.ndarray): Input data of shape (m, nx)
        m is the number of data points
        nx is the number of features in X
    Y (numpy.ndarray): Labels of shape (m, ny)
        m is the same number of data points as in X
        ny is the number of classes for classification tasks
    batch_size (int): The number of data points in a batch

    Returns:
    list: A list of mini-batches, each a tuple (X_batch, Y_batch)
    """
    # Shuffle the data and labels consistently
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    mini_batches = []
    m = X.shape[0]

    # Create mini-batches of the specified batch size
    for i in range(0, m, batch_size):
        # Handle the last mini-batch, which may be smaller than batch_size
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches

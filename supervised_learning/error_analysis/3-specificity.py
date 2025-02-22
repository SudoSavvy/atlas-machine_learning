#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (numpy.ndarray): One-hot encoded array of shape (m, classes)
                                with correct labels.
        logits (numpy.ndarray): One-hot encoded array of shape (m, classes)
                                with predicted labels.

    Returns:
        numpy.ndarray: A confusion matrix of shape (classes, classes) where rows
                       represent actual labels and columns represent predictions.
    """
    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=float)
    for i in range(len(actual)):
        confusion[actual[i], predicted[i]] += 1.0
    return confusion


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A confusion matrix of shape (classes, classes),
                                   where rows represent actual labels and
                                   columns represent predictions.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the sensitivity
                       of each class.
    """
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    return true_positives / (true_positives + false_negatives)


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A confusion matrix of shape (classes, classes),
                                   where rows represent actual labels and
                                   columns represent predictions.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the precision
                       of each class.
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    return true_positives / (true_positives + false_positives)


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A confusion matrix of shape (classes, classes),
                                   where rows represent actual labels and
                                   columns represent predictions.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the specificity
                       of each class.
    """
    true_negatives = np.sum(confusion) - (
        np.sum(confusion, axis=0) + np.sum(confusion, axis=1) - np.diag(confusion)
    )
    false_positives = np.sum(confusion, axis=0) - np.diag(confusion)
    return true_negatives / (true_negatives + false_positives)

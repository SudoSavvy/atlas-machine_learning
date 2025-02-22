#!/usr/bin/env python3
import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (numpy.ndarray): One-hot encoded array of shape (m, classes) with correct labels.
        logits (numpy.ndarray): One-hot encoded array of shape (m, classes) with predicted labels.

    Returns:
        numpy.ndarray: A confusion matrix of shape (classes, classes) where rows
                       represent actual labels and columns represent predictions.
    """
    # Convert one-hot encoding to class indices
    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)
    
    # Get number of classes
    classes = labels.shape[1]
    
    # Initialize confusion matrix
    confusion = np.zeros((classes, classes), dtype=float)
    
    # Populate confusion matrix
    for i in range(len(actual)):
        confusion[actual[i], predicted[i]] += 1.0
    
    return confusion

def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   where rows represent actual labels and columns
                                   represent predicted labels.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the sensitivity
                       of each class.
    """
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    sensitivity_scores = true_positives / (true_positives + false_negatives)
    
    return sensitivity_scores

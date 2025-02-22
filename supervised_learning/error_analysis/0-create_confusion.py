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
    confusion = np.zeros((classes, classes), dtype=int)
    
    # Populate confusion matrix
    for i in range(len(actual)):
        confusion[actual[i], predicted[i]] += 1
    
    return confusion

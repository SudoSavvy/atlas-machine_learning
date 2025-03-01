#!/usr/bin/env python3
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters:
    - network: The model to train
    - data: Numpy array of shape (m, nx) containing the input data
    - labels: One-hot numpy array of shape (m, classes) containing the labels of data
    - batch_size: Size of the batch used for mini-batch gradient descent
    - epochs: Number of passes through data for mini-batch gradient descent
    - verbose: Boolean to determine if output should be printed during training
    - shuffle: Boolean to determine whether to shuffle the batches every epoch

    Returns:
    - The History object generated after training the model
    """
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)

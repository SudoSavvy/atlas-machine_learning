#!/usr/bin/env python3
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and optionally analyzes validation data.
    Supports early stopping based on validation loss.

    Parameters:
    - network: The model to train
    - data: Numpy array of shape (m, nx) containing the input data
    - labels: One-hot numpy array of shape (m, classes) containing the labels of data
    - batch_size: Size of the batch used for mini-batch gradient descent
    - epochs: Number of passes through data for mini-batch gradient descent
    - validation_data: Tuple (val_data, val_labels) for model validation, if not None
    - early_stopping: Boolean to determine whether early stopping should be used
      (only applied if validation_data is provided)
    - patience: Number of epochs with no improvement after which training stops
    - verbose: Boolean to determine if output should be printed during training
    - shuffle: Boolean to determine whether to shuffle the batches every epoch

    Returns:
    - The History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        callbacks.append(early_stop)

    return network.fit(data, labels, batch_size=batch_size, epochs=epochs, 
                       validation_data=validation_data, verbose=verbose,
                       shuffle=shuffle, callbacks=callbacks)

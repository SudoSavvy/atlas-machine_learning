#!/usr/bin/env python3
import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and optionally analyzes validation data.
    Supports early stopping based on validation loss and learning rate decay.

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
    - learning_rate_decay: Boolean to determine whether learning rate decay should be used
      (only applied if validation_data is provided)
    - alpha: Initial learning rate
    - decay_rate: Decay rate for inverse time decay
    - verbose: Boolean to determine if output should be printed during training
    - shuffle: Boolean to determine whether to shuffle the batches every epoch

    Returns:
    - The History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        callbacks.append(early_stop)
    
    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)
        lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay)
    
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs, 
                       validation_data=validation_data, verbose=verbose, 
                       shuffle=shuffle, callbacks=callbacks)

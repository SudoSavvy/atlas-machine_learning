#!/usr/bin/env python3

"""
Module for calculating accuracy, loss, training operation, training, and evaluating a neural network classifier using TensorFlow v1.
"""

import tensorflow.compat.v1 as tf

def calculate_accuracy(y, y_pred):
    """
    Compute the accuracy of a prediction.

    Args:
        y (tf.Tensor): Placeholder tensor containing the true labels.
        y_pred (tf.Tensor): Tensor containing the network’s predictions.

    Returns:
        tf.Tensor: A tensor representing the decimal accuracy of the prediction.
    """
    correct_predictions = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def calculate_loss(y, y_pred):
    """
    Compute the softmax cross-entropy loss of a prediction.

    Args:
        y (tf.Tensor): Placeholder tensor containing the true labels.
        y_pred (tf.Tensor): Tensor containing the network’s predictions.

    Returns:
        tf.Tensor: A tensor representing the loss of the prediction.
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))
    return loss

def create_train_op(loss, alpha):
    """
    Create the training operation for the network using gradient descent.

    Args:
        loss (tf.Tensor): Tensor representing the loss of the network’s prediction.
        alpha (float): Learning rate for the optimizer.

    Returns:
        tf.Operation: An operation that trains the network using gradient descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op

def evaluate(X, Y, save_path):
    """
    Evaluate the output of a neural network.

    Args:
        X (np.ndarray): Input data to evaluate.
        Y (np.ndarray): One-hot labels for X.
        save_path (str): Location to load the model from.

    Returns:
        tuple: The network’s prediction, accuracy, and loss.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        
        prediction = sess.run(y_pred, feed_dict={x: X})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        cost = sess.run(loss, feed_dict={x: X, y: Y})
        
    return prediction, acc, cost

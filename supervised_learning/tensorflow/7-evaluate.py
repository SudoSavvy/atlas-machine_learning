#!/usr/bin/env python3

"""
Module for calculating accuracy, loss, training operation, training, and evaluating a neural network classifier using TensorFlow v1.
"""

import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op

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

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Build, train, and save a neural network classifier.

    Args:
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training labels.
        X_valid (np.ndarray): Validation input data.
        Y_valid (np.ndarray): Validation labels.
        layer_sizes (list): Number of nodes in each layer of the network.
        activations (list): Activation functions for each layer of the network.
        alpha (float): Learning rate.
        iterations (int): Number of iterations to train over.
        save_path (str): Path to save the model.

    Returns:
        str: The path where the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            train_cost, train_acc = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")
            
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        
        save_path = saver.save(sess, save_path)
    return save_path

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
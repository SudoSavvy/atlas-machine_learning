#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using TensorFlow.
    :param x: tf.placeholder of shape (m, 28, 28, 1) containing the input images
    :param y: tf.placeholder of shape (m, 10) containing the one-hot labels
    :return: softmax output tensor, training operation, loss tensor, accuracy tensor
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional Layer 1
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same',
                             activation=tf.nn.relu, kernel_initializer=initializer)
    # Max Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, padding='valid',
                             activation=tf.nn.relu, kernel_initializer=initializer)
    # Max Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Flatten Layer
    flat = tf.layers.flatten(pool2)

    # Fully Connected Layer 1
    fc1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # Fully Connected Layer 2
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # Output Layer with Softmax Activation
    logits = tf.layers.dense(fc2, units=10, kernel_initializer=initializer)
    softmax = tf.nn.softmax(logits)

    # Loss Function (Cross-Entropy)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

    # Training Operation (Adam Optimizer)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy Calculation
    correct_preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    return softmax, train_op, loss, accuracy

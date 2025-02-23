#!/usr/bin/env python3
"""L2 Regularization Cost Module"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Function that calculates the cost of a neural network
    with L2 regularization:

    cost = a tensor containing the cost of the network without L2
    regularization
    model = a Keras model that includes layers with L2 regularization

    """
    # creates an empty list to store the costs
    l2_costs = []
    # add the l2 losses from each layer to the cost
    for l2_loss in model.losses:
        l2_costs.append(cost + l2_loss)
    # converts the list of layers into one tensor
    tensor_cost = tf.convert_to_tensor(l2_costs)

    # returns a tensor containing the total cost for each layer with L2 reg
    return (tensor_cost)

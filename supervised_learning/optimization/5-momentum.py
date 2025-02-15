#!/usr/bin/env python3

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization algorithm.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight.
    var (numpy.ndarray or float): The variable to be updated.
    grad (numpy.ndarray or float): The gradient of var.
    v (numpy.ndarray or float): The previous first moment of var.

    Returns:
    tuple: The updated variable and the new moment.
    """
    v = beta1 * v + (1 - beta1) * grad  # Compute new momentum
    var = var - alpha * v  # Update variable

    return var, v

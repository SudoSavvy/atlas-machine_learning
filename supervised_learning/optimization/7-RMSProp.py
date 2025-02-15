#!/usr/bin/env python3

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    alpha (float): Learning rate.
    beta2 (float): RMSProp weight (decay rate).
    epsilon (float): Small number to avoid division by zero.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of the variable.
    s (numpy.ndarray): Previous second moment of the variable.

    Returns:
    tuple: Updated variable and the new second moment.
    """
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    # Update second moment estimate
    var_new = var - alpha * grad / (np.sqrt(s_new) + epsilon)
    # Update variable

    return var_new, s_new

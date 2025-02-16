#!/usr/bin/env python3

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Parameters:
    alpha (float): Learning rate.
    beta1 (float): Weight for the first moment.
    beta2 (float): Weight for the second moment.
    epsilon (float): Small number to avoid division by zero.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of the variable.
    v (numpy.ndarray): Previous first moment of the variable.
    s (numpy.ndarray): Previous second moment of the variable.
    t (int): Time step used for bias correction.

    Returns:
    tuple: Updated variable, new first moment, and new second moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    var -= alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s

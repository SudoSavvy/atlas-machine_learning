#!/usr/bin/env python3

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Parameters:
    alpha (float): Original learning rate.
    decay_rate (float): Weight to determine the rate of decay.
    global_step (int): Number of passes of gradient descent that have elapsed.
    decay_step (int): Number of passes before learning rate is decayed further

    Returns:
    float: Updated learning rate.
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))

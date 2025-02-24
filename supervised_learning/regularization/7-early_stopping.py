#!/usr/bin/env python3
"""Early Stopping Module"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Function that determines if you should stop gradient descent early:

    Early stopping should occur when the validation cost of the network
    has not decreased relative to the optimal validation cost by more than
    the threshold over a specific patience count
    -> cost = the current validation cost of the neural network
    -> opt_cost = the lowest recorded validation cost of the neural network
    -> threshold = the threshold used for early stopping
    -> patience = the patience count used for early stopping
    -> count = the count of how long the threshold has not been met

    """
    # reset the count if the cost improve, alternatively increment
    count = 0 if cost < opt_cost - threshold else count + 1

    # returns a boolean, followed by the updated count
    return (count >= patience), count

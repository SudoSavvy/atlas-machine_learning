#!/usr/bin/env python3
import numpy as np


def moving_average(data, bea):
    """
    Calculates the weighted moving average of a data set using bias correction.
    
    Parameters:
    data (list): The list of data to calculate the moving average of.
    beta (float): The weight used for the moving average.
    
    Returns:
    list: A list containing the moving averages of data.
    """
    v = 0
    moving_averages = []

    for t in range(1, len(data) + 1):
        v = beta * v + (1 - beta) * data[t -1]
        bias_correction = 1 - beta**t
        moving_averages.append(v / bias_correction)

    return moving_averages

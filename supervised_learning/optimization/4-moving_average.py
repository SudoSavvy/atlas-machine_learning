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
    v = 0.0  # Initialize moving average variable
    moving_averages = []  # List to store moving average values

    for t in range(1, len(data) + 1):
        v = beta * v + (1 - beta) * data[t - 1]  # Compute weighted moving average
        bias_correction = 1 - beta**t  # Compute bias correction factor
        moving_averages.append(v / bias_correction)  # Apply correction and store result

    return moving_averages

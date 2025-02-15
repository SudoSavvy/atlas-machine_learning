#!/usr/bin/env python3

import numpy as np


def moving_average(data, beta):
    """
    Calculates the exponentially weighted moving average (EWMA) of a dataset
    using bias correction.

    Parameters:
    data (list): List of numerical values to compute the moving average of.
    beta (float): Weight factor (0 < beta < 1) determining how much
                  previous values influence the moving average.

    Returns:
    list: A list of the computed moving averages for each time step.
    """
    v = 0.0  # Initialize moving average variable
    moving_averages = []  # List to store moving average values

    for t in range(1, len(data) + 1):
        v = beta * v + (1 - beta) * data[t - 1]  
        # Compute weighted moving average
        bias_correction = 1 - beta**t  
        # Compute bias correction factor
        moving_averages.append(v / bias_correction)  
        # Apply correction and store result

    return moving_averages

# Exact input dataset to match expected output
if __name__ == "__main__":
    data = [
        145.0, 149.0, 173.0, 98.0, 99.0, 76.0, 69.0, 77.0, 79.0, 97.0,
        72.0, 81.0, 79.0, 47.0, 99.0, 49.0, 97.0, 18.0, 59.0, 86.0,
        111.0, 158.0, 69.0, 135.0, 54.0, 141.0, 116.0, 99.0, 66.0,
        38.0, 125.0, 64.0, 94.0, 39.0, 99.0, 92.0, 119.0, 54.0, 65.0,
        24.0, 82.0, 48.0, 105.0, 72.0, 29.0, 36.0, 86.0, 67.0, 79.0,
        102.0, 76.0, 33.0, 83.0, 37.0, 59.0, 94.0
    ]
    beta = 0.9  # Given weight
    result = moving_average(data, beta)
    
    # Ensure the exact formatting of the desired output
    print(result)

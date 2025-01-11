#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def scatter():
    # Generate the data
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)  # Set seed for reproducibility
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180  # Adjust weight data

    # Create the scatter plot
    plt.scatter(x, y, c='magenta', s=10)  # magenta points, size 10

    # Set the title and labels
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    # Set the axis limits
    plt.xlim(55, 80)
    plt.ylim(165, 195)

    # Display the plot
    plt.show()

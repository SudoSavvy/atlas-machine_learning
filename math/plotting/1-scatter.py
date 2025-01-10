#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def scatter():
    # Generate data using the specified mean and covariance
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    # Create the scatter plot
    plt.scatter(x, y, color='magenta', s=10)

    # Set the title, x-axis label, and y-axis label
    plt.title("Men's Height vs Weight", fontsize=14)
    plt.xlabel("Height (in)", fontsize=12)
    plt.ylabel("Weight (lbs)", fontsize=12)

    # Adjust plot limits to match reference
    plt.xlim(55, 80)
    plt.ylim(165, 195)

    # Display the plot
    plt.show()

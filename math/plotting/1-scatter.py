#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def scatter():
    # Generate the data
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    # Create the scatter plot
    plt.scatter(x, y, color='magenta', s=10)

    # Set the title and labels
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    # Fix the axis limits (may be required for exact match)
    plt.xlim(55, 80)
    plt.ylim(165, 195)

    # Standardize the layout to avoid issues
    plt.tight_layout()

    # Show the plot
    plt.show()

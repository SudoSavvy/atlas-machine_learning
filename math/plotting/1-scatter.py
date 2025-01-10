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

    # Create the scatter plot with the correct marker size and color
    plt.scatter(x, y, c='magenta', s=10)  # Match point size and color

    # Set the title and labels
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    # Set exact axis limits
    plt.xlim(55, 80)
    plt.ylim(165, 195)

    # Use a default figure size to avoid scaling issues
    plt.gcf().set_size_inches(6.4, 4.8)

    # Display the plot
    plt.show()

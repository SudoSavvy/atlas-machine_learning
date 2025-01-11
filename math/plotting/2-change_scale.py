#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    # Generate the data
    x = np.arange(0, 28651, 5730)  # Time in years
    r = np.log(0.5)  # Decay rate
    t = 5730  # Half-life of C-14
    y = np.exp((r / t) * x)  # Exponential decay equation

    # Create the plot
    plt.plot(x, y)  # Plot the data as a line graph

    # Set labels and title
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    # Set the Y-axis to a logarithmic scale
    plt.yscale('log')

    # Set the X-axis range
    plt.xlim(0, 28650)

    # Display the plot
    plt.show()

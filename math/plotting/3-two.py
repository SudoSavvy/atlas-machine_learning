#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def two():
    # Generate data
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)  # Decay constant
    t1 = 5730  # Half-life of C-14
    t2 = 1600  # Half-life of Ra-226
    y1 = np.exp((r / t1) * x)  # Exponential decay for C-14
    y2 = np.exp((r / t2) * x)  # Exponential decay for Ra-226

    # Create the plot
    plt.plot(x, y1, 'r--', label='C-14')  # Dashed red line for C-14
    plt.plot(x, y2, 'g-', label='Ra-226')  # Solid green line for Ra-226

    # Add labels, title, and legend
    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    # Set axis ranges
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Add legend in the upper right corner
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()

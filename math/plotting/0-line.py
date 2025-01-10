#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    y = np.arange(0, 11) ** 3
    plt.plot(range(11), y, 'r-')  # Solid red line
    plt.xlim(0, 10)  # Set x-axis range
    plt.show()

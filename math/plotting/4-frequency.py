#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Plot the histogram with the exact specifications
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black', align='mid')
    
    # Add labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Set exact limits to match the example
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # Show the plot
    plt.show()

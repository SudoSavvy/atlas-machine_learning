#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    # Set random seed for reproducibility
    np.random.seed(5)

    # Generate the grades
    student_grades = np.random.normal(68, 15, 50)

    # Create the histogram with appropriate bins and black outlines for bars
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black', linewidth=1.2)

    # Add the required labels and title
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.title('Project A', fontsize='x-small')

    # Adjust the ticks and layout for better appearance
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')

    # Display the plot
    plt.show()

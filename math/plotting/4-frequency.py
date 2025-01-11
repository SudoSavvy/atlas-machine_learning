#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    # Set the random seed for reproducibility
    np.random.seed(5)

    # Generate student grades
    student_grades = np.random.normal(68, 15, 50)

    # Create the histogram
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Add axis labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Set x-axis ticks and layout for better clarity
    plt.xticks(range(0, 101, 10))

    # Display the plot
    plt.show()

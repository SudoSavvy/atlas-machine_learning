#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def all_in_one():

    # Data for the plots
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Create the figure and the 3x2 grid
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('All in One', fontsize='x-small')

    # Top-left: y = x^3
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(np.arange(0, 11), y0, color='red')
    ax1.set_title('y = x^3', fontsize='x-small')

    # Top-right: Scatter plot
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(x1, y1, color='magenta', s=1)
    ax2.set_title("Men's Height vs Weight", fontsize='x-small')
    ax2.set_xlabel('Height (in)', fontsize='x-small')
    ax2.set_ylabel('Weight (lbs)', fontsize='x-small')

    # Middle-left: Exponential decay of C-14
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(x2, y2)
    ax3.set_title('Exponential Decay of C-14', fontsize='x-small')
    ax3.set_xlabel('Time (years)', fontsize='x-small')
    ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax3.set_yscale('log')

    # Middle-right: Exponential decay of Ra-226 and C-14
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(x3, y31, 'r--', label='C-14')
    ax4.plot(x3, y32, 'g-', label='Ra-226')
    ax4.set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    ax4.set_xlabel('Time (years)', fontsize='x-small')
    ax4.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax4.legend(fontsize='x-small')

    # Bottom: Histogram of grades spanning two columns
    ax5 = plt.subplot(3, 2, (5, 6))
    ax5.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    ax5.set_title('Project A', fontsize='x-small')
    ax5.set_xlabel('Grades', fontsize='x-small')
    ax5.set_ylabel('Number of Students', fontsize='x-small')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Leave space for the main title

    # Show the plots
    plt.show()

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    
    # Data
    people = ['Farrah', 'Fred', 'Felicia']
    apples, bananas, oranges, peaches = fruit

    # Bar positions
    x = np.arange(len(people))

    # Create the plot
    plt.bar(x, apples, width=0.5, color='red', label='apples')
    plt.bar(x, bananas, width=0.5, bottom=apples, color='yellow', label='bananas')
    plt.bar(x, oranges, width=0.5, bottom=apples + bananas, color='#ff8000', label='oranges')
    plt.bar(x, peaches, width=0.5, bottom=apples + bananas + oranges, color='#ffe5b4', label='peaches')

    # Add labels, ticks, and title
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, people)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()

    # Show plot
    plt.show()

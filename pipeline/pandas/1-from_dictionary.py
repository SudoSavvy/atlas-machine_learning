#!/usr/bin/env python3
"""
This script creates a pandas DataFrame from a dictionary.
The DataFrame contains two columns labeled 'First' and 'Second',
with custom row labels A through D.
"""

import pandas as pd

# Define the data dictionary
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Define the row labels
row_labels = ['A', 'B', 'C', 'D']

# Create the DataFrame
df = pd.DataFrame(data, index=row_labels)

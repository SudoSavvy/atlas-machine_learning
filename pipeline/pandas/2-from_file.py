#!/usr/bin/env python3
"""
This module defines a function that loads data from a file
into a pandas DataFrame using a specified delimiter.
"""

import pandas as pd

def from_file(filename, delimiter):
    """
    Loads data from a file into a pandas DataFrame.

    Parameters:
    filename (str): The path to the file to load.
    delimiter (str): The column separator used in the file.

    Returns:
    pd.DataFrame: The loaded DataFrame containing the file data.
    """
    return pd.read_csv(filename, delimiter=delimiter)

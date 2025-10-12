#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
    """
    Creates a pandas DataFrame from a NumPy ndarray.

    Parameters:
    array (np.ndarray): A NumPy array from which to create the DataFrame.
                        The array should be 2-dimensional.

    Returns:
    pd.DataFrame: A DataFrame with columns labeled in alphabetical order,
                  capitalized from 'A' to 'Z'. Assumes no more than 26 columns.
    """
    num_columns = array.shape[1]
    column_labels = [chr(65 + i) for i in range(num_columns)]
    return pd.DataFrame(array, columns=column_labels)

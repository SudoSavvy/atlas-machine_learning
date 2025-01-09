#!/usr/bin/env python3

# Define the array
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]

# Extract the first two numbers (indices 0 and 1)
arr1 = arr[:2]  # Start from the beginning up to (but not including) index 2

# Extract the last five numbers (indices -5 to the end)
arr2 = arr[-5:]  # Start at index -5 and go to the end

# Extract the 2nd through 6th numbers (indices 1 to 5)
arr3 = arr[1:6]  # Start at index 1 and stop before index 6

# Print the results
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))

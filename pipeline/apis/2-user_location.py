#!/usr/bin/env python3
"""
This script retrieves and prints the location of a GitHub user
using the GitHub API. It handles user not found and rate limit errors.
"""

import sys
import requests
from datetime import datetime


def get_user_location(url):
    """
    Fetch the location of a GitHub user from the provided API URL.

    Parameters:
        url (str): The full GitHub API URL for the user.

    Returns:
        str: The location string, 'Not found' if user doesn't exist,
             or rate limit message if access is restricted.
    """
    response = requests.get(url)
    if response.status_code == 404:
        return 'Not found'
    if response.status_code == 403:
        reset_time = response.headers.get('X-Ratelimit-Reset')
        if reset_time:
            reset_timestamp = int(reset_time)
            now = int(datetime.now().timestamp())
            minutes = (reset_timestamp - now) // 60
            return f'Reset in {minutes} min'
        return 'Reset time unknown'
    if response.status_code == 200:
        data = response.json()
        return data.get('location', 'No location provided')
    return 'Unexpected error'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    url = sys.argv[1]
    print(get_user_location(url))

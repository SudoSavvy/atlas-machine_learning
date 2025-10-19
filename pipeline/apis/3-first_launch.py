#!/usr/bin/env python3
"""
This script retrieves and displays the first SpaceX launch using the unofficial SpaceX API.

It prints:
- Launch name
- Local date and time
- Rocket name
- Launchpad name and locality

Format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests
from datetime import datetime


def get_first_launch():
    """
    Fetch and display the first SpaceX launch with formatted details.

    Returns:
        str: Formatted string with launch name, date, rocket, and launchpad info.
    """
    launches_url = 'https://api.spacexdata.com/v4/launches'
    rockets_url = 'https://api.spacexdata.com/v4/rockets/'
    launchpads_url = 'https://api.spacexdata.com/v4/launchpads/'

    response = requests.get(launches_url)
    if response.status_code != 200:
        return 'Failed to retrieve launches.'

    launches = response.json()
    # Sort by date_unix ascending
    launches.sort(key=lambda x: x.get('date_unix', float('inf')))

    first = launches[0]
    launch_name = first.get('name')
    date_unix = first.get('date_unix')
    rocket_id = first.get('rocket')
    launchpad_id = first.get('launchpad')

    # Convert date_unix to local datetime string
    local_date = datetime.fromtimestamp(date_unix).strftime('%Y-%m-%d %H:%M:%S')

    # Get rocket name
    rocket_response = requests.get(rockets_url + rocket_id)
    rocket_name = rocket_response.json().get('name') if rocket_response.status_code == 200 else 'Unknown Rocket'

    # Get launchpad name and locality
    pad_response = requests.get(launchpads_url + launchpad_id)
    if pad_response.status_code == 200:
        pad_data = pad_response.json()
        pad_name = pad_data.get('name', 'Unknown Pad')
        pad_locality = pad_data.get('locality', 'Unknown Location')
    else:
        pad_name = 'Unknown Pad'
        pad_locality = 'Unknown Location'

    return f"{launch_name} ({local_date}) {rocket_name} - {pad_name} ({pad_locality})"

if __name__ == '__main__':
    print(get_first_launch())

#!/usr/bin/env python3
"""
This script retrieves and displays the first SpaceX launch from the API response.

It prints:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests
from datetime import datetime
import pytz


def get_first_launch():
    """
    Fetch and format the first SpaceX launch from the API response.

    Returns:
        str: Formatted launch string with name, local date, rocket, and launchpad.
    """
    launches_url = 'https://api.spacexdata.com/v4/launches'
    rockets_url = 'https://api.spacexdata.com/v4/rockets/'
    launchpads_url = 'https://api.spacexdata.com/v4/launchpads/'

    response = requests.get(launches_url)
    if response.status_code != 200:
        return 'Failed to retrieve launches.'

    launches = response.json()
    first = launches[0]

    launch_name = first.get('name')
    date_utc = first.get('date_utc')
    rocket_id = first.get('rocket')
    launchpad_id = first.get('launchpad')

    # Convert UTC date to local time with timezone offset
    utc_dt = datetime.fromisoformat(date_utc.replace('Z', '+00:00'))
    local_dt = utc_dt.astimezone()
    formatted_date = local_dt.isoformat()

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

    return f"{launch_name} ({formatted_date}) {rocket_name} - {pad_name} ({pad_locality})"

if __name__ == '__main__':
    print(get_first_launch())

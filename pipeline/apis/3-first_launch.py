#!/usr/bin/env python3
"""
This script retrieves the first listed SpaceX launch from the API
and prints its name, local date, rocket, and launchpad details.
"""

import requests

def fetch_first_launch():
    """
    Fetches the first launch from the SpaceX API response and formats its details.

    Returns:
        str: Formatted string with launch name, date, rocket name, and launchpad info.
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(launches_url)
    launches = response.json()

    first_launch = launches[0]  # Do not sort — use first in API response

    rocket_id = first_launch["rocket"]
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)
    rocket_name = rocket_response.json()["name"]

    launchpad_id = first_launch["launchpad"]
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data["name"]
    launchpad_locality = launchpad_data["locality"]

    launch_name = first_launch["name"]
    launch_date = first_launch["date_local"]

    return f"{launch_name} ({launch_date}) {rocket_name} - {launchpad_name} ({launchpad_locality})"

if __name__ == "__main__":
    print(fetch_first_launch())

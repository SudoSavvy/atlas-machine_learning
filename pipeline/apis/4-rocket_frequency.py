#!/usr/bin/env python3
"""
This script retrieves all SpaceX launches and counts how many times each rocket was used.
It prints each rocket name followed by the number of launches in descending order.
If multiple rockets have the same count, they are sorted alphabetically.
"""

import requests
from collections import defaultdict


def count_launches_per_rocket():
    """
    Fetches launch and rocket data from the SpaceX API and counts launches per rocket.

    Returns:
        list of tuples: Each tuple contains (rocket_name, launch_count), sorted as specified.
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    launches_response = requests.get(launches_url)
    rockets_response = requests.get(rockets_url)

    if launches_response.status_code != 200 or rockets_response.status_code != 200:
        return []

    launches = launches_response.json()
    rockets = rockets_response.json()

    # Map rocket ID to name
    rocket_id_to_name = {rocket["id"]: rocket["name"] for rocket in rockets}

    # Count launches per rocket ID
    launch_counts = defaultdict(int)
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            launch_counts[rocket_id] += 1

    # Convert to list of (name, count) and sort
    result = [
        (rocket_id_to_name.get(rid, "Unknown Rocket"), count)
        for rid, count in launch_counts.items()
    ]
    result.sort(key=lambda x: (-x[1], x[0]))

    return result


if __name__ == "__main__":
    for rocket_name, count in count_launches_per_rocket():
        print(f"{rocket_name}: {count}")

#!/usr/bin/env python3

"""This script retrieves and displays the first SpaceX launch using the public API."""

import requests


def fetch_first_launch():
    """Fetches the first launch from the SpaceX API and formats its details."""

    launches_url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(launches_url)
    launches = response.json()

    launches.sort(key=lambda x: x["date_unix"])

    #   debug prints I gave up on half way in
    # print("5 launches after sorting:")
    # for i in len(launches[:5]):
    #     first_launch = launches[launch]
    #     launches_name = first_launch["name"]
    #     launches_date = launches["date_unix"]
    #     print(f"{launch_name} ({launches_date})")

    first_launch = launches[0]
    # launches_name = first_launch["name"]
    # print("first try")
    # print(f"First launch: {launches_name}")

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
    output = f"{launch_name} ({launch_date}) {rocket_name} - {launchpad_name} ({launchpad_locality})"

    return output


if __name__ == "__main__":
    """Executes the script and prints the formatted launch details."""
    result = fetch_first_launch()
    print(result)

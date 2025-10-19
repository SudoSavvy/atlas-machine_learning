#!/usr/bin/env python3
"""
This module defines a function to retrieve starships from the SWAPI
that can accommodate a given number of passengers.
"""

import requests

def availableShips(passengerCount):
    """
    Retrieve a list of starships from the SWAPI that can hold at least
    the specified number of passengers.

    Parameters:
        passengerCount (int): The minimum number of passengers the ship must support.

    Returns:
        list: A list of ship names that meet the passenger requirement.
    """
    url = 'https://swapi.dev/api/starships/'
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        for ship in data.get('results', []):
            passengers = ship.get('passengers', '0').replace(',', '').replace('n/a', '0').replace('unknown', '0')
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship.get('name'))
            except ValueError:
                continue

        url = data.get('next')

    return ships

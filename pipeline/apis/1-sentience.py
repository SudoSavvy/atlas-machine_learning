#!/usr/bin/env python3
"""
This module defines a function to retrieve the names of home planets
for all sentient species listed in the Star Wars API (SWAPI).
"""

import requests

def sentientPlanets():
    """
    Retrieve the list of home planet names for all sentient species
    from the SWAPI. Sentient classification is determined by either
    the 'classification' or 'designation' attributes.

    Returns:
        list: A list of unique home planet names for sentient species.
    """
    url = 'https://swapi.dev/api/species/'
    planets = set()

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        for species in data.get('results', []):
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()
            if 'sentient' in classification or 'sentient' in designation:
                homeworld_url = species.get('homeworld')
                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        name = planet_data.get('name')
                        if name:
                            planets.add(name)

        url = data.get('next')

    return list(planets)

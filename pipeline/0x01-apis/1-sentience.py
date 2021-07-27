#!/usr/bin/env python3
""" Where I am? """
import requests


def sentientPlanets():
    """ Method that returns the list of names
        of the home planets of all sentient species
    """

    planets = []
    page = 1
    state = True

    while state:
        url = "https://swapi-api.hbtn.io/api/species/?page=" + str(page)
        speciesRequest = requests.get(url)
        species = speciesRequest.json()
        results = species['results']

        for specie in results:
            if specie['classification'] == 'sentient' or \
               specie['designation'] == 'sentient':
                homeworld = specie['homeworld']
                if homeworld is not None:
                    homeworldRequest = requests.get(specie['homeworld'])
                    planet = homeworldRequest.json()
                    planets.append(planet['name'])

        if species['next'] is None:
            state = False
        page += 1

    return planets

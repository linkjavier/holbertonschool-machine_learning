#!/usr/bin/env python3
""" Can I join? """
import requests


def availableShips(passengerCount):
    """ Method that returns the list of ships
        that can hold a given number of passengers
    """

    ships = []
    page = 1
    state = True
    while state:
        url = "https://swapi-api.hbtn.io/api/starships/?page=" + str(page)
        request = requests.get(url)
        data = request.json()
        results = data['results']

        for ship in results:
            passenger = ship['passengers']
            passenger = passenger.replace(',', "")
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                ships.append(ship['name'])

        if data['next'] is None:
            state = False
        page += 1

    return ships

#!/usr/bin/env python3
""" Rate me is you can! """
import sys
import requests
import time


if __name__ == '__main__':
    """ Script that prints the location of a specific user """

    url = sys.argv[1]
    payload = {'Accept': "application/vnd.github.v3+json"}
    request = requests.get(url, params=payload)

    if request.status_code == 200:
        location = request.json()["location"]
        print(location)

    if request.status_code == 404:
        print("Not found")

    if request.status_code == 403:
        limit = request.headers["X-Ratelimit-Reset"]
        x = (int(limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))

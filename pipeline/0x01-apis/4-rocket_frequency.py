#!/usr/bin/env python3
""" How many by rocket? """

import requests as rq


if __name__ == '__main__':
    """ Script that displays the number of launches per rocket """

    launchesUrl = "https://api.spacexdata.com/v4/launches"
    launchesRequest = rq.get(launchesUrl)
    launches = {}

    for launch in launchesRequest.json():
        if launch['rocket'] not in launches:
            launches[launch['rocket']] = 1
        else:
            launches[launch['rocket']] += 1

    rocketsUrl = "https://api.spacexdata.com/v4/rockets/"
    rocketsRequest = rq.get(rocketsUrl)
    rockets = []

    for rocket in rocketsRequest.json():
        if rocket['id'] in launches:
            rockets.append({'rocket': rocket['name'],
                            'launches': launches[rocket['id']]})
        else:
            continue

    launches = sorted(rockets, key=lambda i: i['rocket'])
    launches = sorted(rockets, key=lambda i: i['launches'], reverse=True)

    for i in launches:
        print("{}: {}".format(i['rocket'], i['launches']))

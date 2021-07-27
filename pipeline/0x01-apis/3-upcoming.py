#!/usr/bin/env python3
""" What will be next? """

import requests


if __name__ == '__main__':
    """ Script that displays the upcoming launch with these information:
            Name of the launch
            The date (in local time)
            The rocket name
            The name (with the locality) of the launchpad
    """
    upcomingLaunchesUrl = "https://api.spacexdata.com/v4/launches/upcoming"
    launchesRequest = requests.get(upcomingLaunchesUrl)
    launches = sorted(launchesRequest.json(), key=lambda i: i['date_unix'])
    dateUnix = launches[0]['date_unix']

    for launch in launchesRequest.json():
        if launch['date_unix'] == dateUnix:
            name = launch['name']
            date = launch['date_local']
            rocketId = launch['rocket']
            launchpadId = launch['launchpad']

    rocketUrl = "https://api.spacexdata.com/v4/rockets/{}".format(rocketId)
    rocketRequest = requests.get(rocketUrl)
    rocketName = rocketRequest.json()['name']

    launchpadUrl = "https://api.spacexdata.com/v4/launchpads/{}".format(
        launchpadId)
    launchpadRequest = requests.get(launchpadUrl)
    launchpadName = launchpadRequest.json()['name']
    launchpadLocality = launchpadRequest.json()['locality']

    print("{} ({}) {} - {} ({})".format(name, date, rocketName,
                                        launchpadName, launchpadLocality))

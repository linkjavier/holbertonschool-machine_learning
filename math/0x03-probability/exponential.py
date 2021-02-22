#!/usr/bin/env python3
""" Initialize Exponential PDF CDF"""


class Exponential:
    """ Class Exponential that represents an exponential distribution """

    EULER = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""

        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1/(sum(data)/len(data))

        else:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """ Calculates the value of the PDF for a given time period """

        if x < 0:
            return 0

        return self.lambtha * (self.EULER ** (-self.lambtha * x))

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period """

        if x < 0:
            return 0

        return 1 - (self.EULER ** (-self.lambtha * x))

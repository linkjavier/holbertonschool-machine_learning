#!/usr/bin/env python3
""" Initialize Binomial """


class Binomial:
    """Class Binomial that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""
        if data is not None:
            n, p = self.calculate_n_p(data)

        self.n = n
        self.p = p

    @property
    def n(self):
        """ Bernoulli trials getter"""

        return self.__n

    @n.setter
    def n(self, n):
        """ Bernoulli trials setter"""

        if n <= 0:
            raise ValueError('n must be a positive value')
        self.__n = int(n)

    @property
    def p(self):
        """ Probability getter"""

        return self.__p

    @p.setter
    def p(self, p):
        """ Probability setter"""

        if not 0 < p < 1:
            raise ValueError('p must be greater than 0 and less than 1')
        self.__p = float(p)

    @staticmethod
    def factorial(n):
        """ Calculates factorial """
        fact = 1
        for i in range(1, n + 1):
            fact *= i
        return fact

    @classmethod
    def calculate_n_p(cls, data):
        """Calculates n Bernoulli trials and probability"""
        if not isinstance(data, list):
            raise TypeError('data must be a list')
        if len(data) < 2:
            raise ValueError('data must contain multiple values')

        DataLength = len(data)
        mean = sum(data)/DataLength
        variance = sum([(number - mean) ** 2 for number in data])/DataLength
        p = 1 - (variance/mean)
        n = int(round(mean/p))
        p = (mean/n)
        return n, p

    def pmf(self, k):
        """ Calculates the value of the PMF for
            a given number of successes
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        binomial = self.get_bcf(k)
        q = 1 - self.p
        return binomial * ((self.p ** k) * (q ** (self.n - k)))

    def cdf(self, k):
        """ Calculates the value of the CDF for
            a given number of successes
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return sum([self.CDF_Function(i) for i in range(k + 1)])

    def get_bcf(self, k):
        """Calculates binomial coefficient with a given number"""
        n_factorial = self.factorial(self.n)
        k_factorial = self.factorial(k)
        n_k_factorial = self.factorial(self.n - k)
        binomial = n_factorial/(n_k_factorial * k_factorial)
        return binomial

    def CDF_Function(self, i):
        """Calculates cdf for each iteration"""
        r = self.get_bcf(i) * ((self.p ** i) * ((1 - self.p) ** (self.n - i)))
        return r

#!/usr/bin/env python3
""" Summation Squared """


def summation_i_squared(n):
    """  Function that calculates sum_{i=1}^{n} i^2 """

    if not isinstance(n, int) or n <= 0:
        return None
    result = n * (n + 1) * (2 * n + 1) // 6
    return result

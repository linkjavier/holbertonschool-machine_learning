#!/usr/bin/env python3
"""Normalization module"""
import numpy as np


def normalize(X, m, s):
    """ Function that normalizes (standardizes) a matrix """
    X -= m
    X /= s
    return X

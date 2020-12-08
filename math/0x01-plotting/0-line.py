#!/usr/bin/env python3
""" Source code to plot y as a line graph """
import numpy as np
import matplotlib.pyplot as plt


y = np.arange(0, 11) ** 3
plt.plot(y, 'r')
plt.show()

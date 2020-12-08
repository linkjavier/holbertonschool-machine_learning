#!/usr/bin/env python3
""" Source code to plot a stacked bar graph """
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

Fruits = ['apples', 'bananas', 'oranges', 'peaches']
Colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
Names = ['Farrah', 'Fred', 'Felicia']
Zeros = np.zeros(3)

for i in range(4):
    plt.bar(Names,
            fruit[i],
            0.5,
            Zeros,
            color=Colors[i],
            label=Fruits[i])

    Zeros += fruit[i]

plt.legend()
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.ytick = 10
plt.axis(ymax=80)
plt.show()

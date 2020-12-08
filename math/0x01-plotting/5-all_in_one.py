#!/usr/bin/env python3
""" Source code to plot 5 graphs in one figure: """
import numpy as np
import matplotlib.pyplot as plt


y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

figurePlot = plt.figure()
plt.rcParams.update({'axes.titlesize': 'x-small',
                     'axes.labelsize': 'x-small'})
figurePlot.add_subplot(3, 2, 1)
plt.plot(y0, 'r')

figurePlot.add_subplot(3, 2, 2)
plt.title("Men's Height vs Weight")
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.scatter(x1, y1, c='m')

figurePlot.add_subplot(3, 2, 3)
plt.title('Exponential Decay of C-14')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.yscale('log')
plt.plot(x2, y2)
plt.axis(xmin=0, xmax=28650)

figurePlot.add_subplot(3, 2, 4)
plt.title('Exponential Decay of Radioactive Elements')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.plot(x3, y31, 'r--', x3, y32, 'g')
plt.axis([0, 20000, 0, 1])
plt.legend(['C-14', 'Ra-226'])

figurePlot.add_subplot(3, 1, 3)
plt.title('Project A')
plt.hist(student_grades, np.arange(0, 101, 10), edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.xticks(np.arange(0, 101, 10))
plt.axis([0, 100, 0, 30])

figurePlot.suptitle('All in One')
figurePlot.tight_layout()
plt.show()

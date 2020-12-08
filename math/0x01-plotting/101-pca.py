#!/usr/bin/env python3
""" Source code to visualize the data in 3D """
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure()

Axes = fig.add_subplot(111, projection='3d')
Axes.set_xlabel('U1')
Axes.set_ylabel('U2')
Axes.set_zlabel('U3')
plt.title('PCA of Iris Dataset')
Axes.scatter(pca_data[:, 0],
             pca_data[:, 1],
             pca_data[:, 2],
             c=labels, cmap='plasma')

plt.show()

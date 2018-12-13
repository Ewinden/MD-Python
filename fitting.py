#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:00:36 2018

@author: alejo
"""
import warnings
warnings.simplefilter("ignore")
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.tri as mtri
import pandas as pd

#read Excel file
df = pd.read_excel('/home/eamon/Downloads/prueba.xlsx')

#arrange data
X1 = df.X.values
Y1 = df.Y.values
Z1 = df.Z.values

X = X1[:, np.newaxis]
Y = Y1[:, np.newaxis]
z = Z1[:, np.newaxis]

xi = np.linspace(min(X), max(X), 8)
yi = np.linspace(min(Y), max(Y), 8)
zi = np.reshape(z, (8, 8))
grid_x, grid_y = np.meshgrid(xi, yi)
#grid_x, grid_y = np.mgrid[:1:100j, 0:1:200j]
points = np.hstack([X, Y])
values = Z1
print(xi)
print(yi)
print(zi)

#fitting
zsmoth = griddata(points, values, (grid_x, grid_y), method='cubic')
fncZ = sp.interpolate.RectBivariateSpline(xi, yi, zi)  #this function can be evaluated in any point

zp = fncZ(70000, 35000)

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf =  ax.contour3D(grid_x, grid_y, zsmoth, 100, cmap=cm.jet)        
                       
#plt.imshow(zsmoth, origin='lower')             

# Customize the z axis.
ax.set_zlim(8980, 9000)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



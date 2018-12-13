'''
Takes images in a specific directory and translates input coordinates into output coordinates

Created on Thu Dec 13 2018

@author: ewinden
'''

import warnings
warnings.simplefilter("ignore")
import cv2
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.tri as mtri

'''Define image locations'''
imagedir = '/home/eamon/Downloads/MDDATA/'
zinputimage = 'ZInputArray.tif'
xinputimage = 'XInputArray.tif'
yinputimage = 'YInputArray.tif'
zoutputimage = 'ZOutputArray.tif'
xoutputimage = 'XOutputArray.tif'
youtputimage = 'YOutputArray.tif'

'''Open images'''
zinputimg = cv2.imread(imagedir+zinputimage,-1)
xinputimg = cv2.imread(imagedir+xinputimage,-1)
yinputimg = cv2.imread(imagedir+yinputimage,-1)
xoutputimg = cv2.imread(imagedir+xoutputimage,-1)
youtputimg = cv2.imread(imagedir+youtputimage,-1)

'''Prepare data for processing, (xyina, zina for griddata) (xinb,yinb for rectbivariatespline)'''
xyina = []
zina = []
xinb = []
yinb = []
c,r = zinputimg.shape
for row in range(c):
	for col in range(r):
		xyina.append([xinputimg[row][col],yinputimg[row][col]])
		zina.append(zinputimg[row][col])

for dta in xinputimg[0]:
	xinb.append(dta)
for dta in yinputimg:
	yinb.append(dta[0])


'''Fit surface and find new Z position outputs'''
zoutput = griddata(xyina, zina, (xoutputimg, youtputimg), method='cubic')
fncZ = sp.interpolate.RectBivariateSpline(yinb, xinb, zinputimg)
rows,cols = xoutputimg.shape
zoutputimg = np.zeros([rows,cols])

for row in range(rows):
	for col in range(cols):
		zoutputimg[row][col]=fncZ(youtputimg[row][col],xoutputimg[row][col])[0][0]

print(str(zinputimg)+'\n')
print(str(zoutputimg)+'\n')
print(str(zoutput)+'\n')


'''Write image'''
# ~ cv2.imwrite(imagedir+zoutputimage,zoutput.astype(np.uint16))

'''Plot surface'''
fig = plt.figure()
ax = fig.gca(projection='3d')
surf =  ax.contour3D(xoutputimg, youtputimg, zoutput, 1000, cmap=cm.jet)           
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.show()

fig2 = plt.figure()
ax = fig2.gca(projection='3d')
surf2 =  ax.contour3D(xoutputimg, youtputimg, zoutputimg, 1000, cmap=cm.jet)           
fig2.colorbar(surf2, shrink=0.5, aspect=5)

fig2.show()
raw_input()


'''
Finds all objects in a folder and fits gaussian curves to each, adding them to a data file.

Created on Thu Dec 13 2018

@author: ewinden
'''

import warnings
warnings.simplefilter("ignore")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
from skimage import morphology
from scipy.misc import toimage
from skimage.measure import regionprops
from skimage.morphology import label
from skimage import feature
from xlrd import open_workbook
import os.path
import imutils
import time
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import pdb
from sys import exit
from gaussfitter import gaussfit
from math import sqrt


def gaussian(x, a, b, c, d): 
	return a*np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2))) + d

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
	pill = abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
	return pill

def detect_name(file_name):
	
	#from wScan checks if a given folder is a good fit
	
	
	if file_name == ():
		print("nothing is chosen,Exit!")
		exit()
	print("The path you chose is " + str(file_name)	)
	
	
def my_convert_16U_2_8U(image):
	
	#from wScan converts a 16 bit image to an 8 bit one using a flat scalar from the lowest to the highest value
	
	min_ = np.amin(image)
	max_ = np.amax(image)
	a = 255/float(max_-min_)
	b = -a*min_
	#print min_, max_ , a, b 
	img8U = np.zeros(image.shape,np.uint8)
	cv2.convertScaleAbs(image,img8U,a,b)
	return img8U


def displayPoint(image, point):
	
	#displays given point in red in an 8bit image
	
	img = cv2.imread(image,1)
	#imblur = cv2.GaussianBlur(img,(5,5),0)
	img8 = my_convert_16U_2_8U(img)
	cv2.line(img8, (point[0],point[1]), (point[0],point[1]),(0,0,255),1)
	cv2.imshow('Window',img8)
	cv2.waitKey(0)
	
	
def drawGaussCircle(img, center, width):
	
	#pull up an image and draws a circle to show the radius determined
	
	h, w = img.shape[:2]
	print(h,w)
	print(center)
	print(width)
	img8 = my_convert_16U_2_8U(img)
	img4x = cv2.resize(img8, (w, h), cv2.INTER_NEAREST)
	center4x = (int(center[0]), int(center[1]))
	cv2.circle(img4x, (center4x[0], center4x[1]), int(width), (0,255,0),1)
	cv2.rectangle(img4x, (center4x[0], center4x[1]),(center4x[0], center4x[1]),(0,0,255),1)
	cv2.imshow('Window',img4x)
	cv2.waitKey(0)
	
	
def getNeighbors(locs):
	
	#Given a list of points, returns a shape around the points of all neighbors
	
	trub = list(locs)
	for pix in locs:
		for x in [pix[0]-1,pix[0],pix[0]+1]:
			for y in [pix[1]-1,pix[1],pix[1]+1]:
				if (x,y) not in trub:
					trub.append((x,y))
	return trub


def checkOnePeak(img):
	
	#Checks if only one peak is in image by distance transform and then counting contours
	
	imb = cv2.GaussianBlur(my_convert_16U_2_8U(img), (5,5), 0)
	ret1, thr1 = cv2.threshold(imb, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	dt = cv2.distanceTransform(thr1, cv2.DIST_L2, 3)
	ret2, thr2 = cv2.threshold(dt.astype('uint8'), 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	contours = cv2.findContours(thr2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	if len(contours[1]) == 1:
		return(True)
	else:
		return(False)


def getBrightBox(img, point, outLength):
	
	#Given a point, returns a square around the point size 2n+1
	shap = img.shape
	startx = int(max(0, point[1] - outLength))
	starty = int(max(0, point[0] - outLength))
	endx = int(min(shap[1], point[1] + outLength+1))
	endy = int(min(shap[0], point[0] + outLength+1))
	brightBox = img[starty:endy,startx:endx]
	avg = brightBox.mean()

	# ~ img8 = my_convert_16U_2_8U(img)
	# ~ cv2.rectangle(img8,(startx,starty),(endx,endy),(255),thickness=1)
	# ~ cv2.line(img8, (point[1],point[0]), (point[1],point[0]),(255,255,255),thickness=1)
	# ~ cv2.imshow('image',img8[starty-50:endy+50,startx-50:endx+50])
	# ~ cv2.waitKey(0)
	# ~ cv2.destroyAllWindows()
	
	# gaussfitter requires float objects in the array passed as data for the fitting
	return((brightBox.astype(float)), startx, starty, avg)


def gaussBoxFitPoint(img, point, outLength):
	
	# ~ Given a point in an image and outLength to test, returns a 2d gaussian fit to the object around the point and its errors
	
	print(point)
	bbout = getBrightBox(img, point, outLength)
	pixSet = bbout[0]
	# ~ sideLength = (outLength*2)+1
	if checkOnePeak(pixSet) == True:
		# ~ xes = np.arange((outLength*2)+1)
		bits = gaussfit((pixSet), return_error='True', circle='True')
		# ~ drawGaussCircle(img, (bbout[2]+bits[0][2], bbout[1]+bits[0][3]), bits[0][4])
		return([bits[0],bits[1],[bbout[3]]])
	else:
		return(np.zeros(5),np.zeros(5),[bbout[3]])

def findBrightObjects(img):
	conts=[]
	shap=img.shape
	img8=my_convert_16U_2_8U(img)
	#img4x = cv2.resize(img8, (shap[1]/2,shap[0]/2), cv2.INTER_NEAREST)
	#imb = cv2.GaussianBlur(img8, (5,5), 0)
	ret1, thr1 = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	# ~ cv2.imshow("img", thr1)
	# ~ cv2.waitKey(0)
	# ~ cv2.destroyAllWindows()
	contours = cv2.findContours(thr1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for cont in contours[1]:
		area = cv2.contourArea(cont)
		per = cv2.arcLength(cont, True)
		(x,y), rad = cv2.minEnclosingCircle(cont)
		conts.append([cont, area,per,x,y,rad])
	return(conts)

def filterContours(conts, areac, perc, xc, yc, radc):
	newConts=[]
	for obj in conts:
		area = obj[1]
		per = obj[2]
		x = obj[3]
		y = obj[4]
		rad = obj[5]
		if area >= areac[0] and area <= areac[1] and per >= perc[0] and per <= perc[1] and x >= xc[0] and x <= xc[1] and y >= yc[0] and y <= yc[1] and rad >= radc[0] and rad <= radc[1]:
			newConts.append(obj) 
	return(newConts)

# ~ print "\n please choose picture!"
# ~ pictouse = askopenfilename(initialdir = "/",title = "Select file")

# ~ print "\n please choose coordinate list from imagej!"
# ~ lsttouse = askopenfilename(title = "Select file")
# ~ directory = (lsttouse[:(lsttouse.rfind('/')+1)])
pictouse = '/home/eamon/Downloads/yuminbeads/mount 1/6.tif'
lsttouse = '/home/eamon/Downloads/yuminbeads/mount 1/6.txt'

newfh = lsttouse[:(lsttouse.rfind('.'))]+'_handanalysis.txt'
newfa = lsttouse[:(lsttouse.rfind('.'))]+'_autoanalysis.txt'
newft = lsttouse[:(lsttouse.rfind('.'))]+'_comparedanalysis.txt'
#print(pictouse)
#print(lsttouse)
#print(directory)
#print(newf)

# ~ pictouse = '/home/eamon/Documents/beads/beads.tif'
# ~ lsttouse = '/home/eamon/Documents/beads/beads.txt'
# ~ newf = '/home/eamon/Documents/beads/analysis.txt'



#Opens files
locsfile = open(lsttouse, 'r')	
img = cv2.imread(pictouse,-1)

outL = 10
shap = img.shape
# ~ intoh = open(newfh, 'w')
# ~ intoa = open(newfa, 'w')
# ~ intoa.write('area, perimeter, radius, x, y, background, peak, xin, yin, width, background error, peak error, x error, y error, width error, average, wnm, pzd\n')
# ~ intoh.write('name, slice, id, xpick, ypick, background, peak, xin, yin, width, background error, peak error, x error, y error, width error, average, wnm, pzd\n')
# ~ locas = findBrightObjects(img)

# ~ #Building table and gaussian analysis for found obects from findBrightObjects
# ~ for line in locas:
	# ~ out = ', '.join(str(o) for o in line)+'\n'
	# ~ area = line[1]
	# ~ per = line[2]
	# ~ x = line[3]
	# ~ y = line[4]
	# ~ pnt = [y,x]
	# ~ rad = line[5]
	# ~ pair = gaussBoxFitPoint(img, pnt, outL)
	# ~ wnm = pair[0][4]*114
	# ~ p2dff = -436.07197+58.30619*sqrt(wnm)
	# ~ pairstr = ''
	# ~ for d in pair:
		# ~ for l in d:
			# ~ pairstr+=', '
			# ~ pairstr+=str(l)
	# ~ out = str(area) + ', ' + str(per) + ', ' + str(rad) + ', ' + str(x) + ', ' + str(y) + pairstr + ', ' + str(wnm) + ', ' + str(p2dff)+'\n'
	# ~ intoa.write(out)
# ~ intoa.close()

# ~ #Building table and gaussian analysis for hand selected objects
# ~ for line in list(locsfile)[1:]:
	# ~ linls = line.split()
	# ~ pointv = linls[0]
	# ~ xpos = int(linls[1])
	# ~ ypos = int(linls[2])
	# ~ slicen = int(linls[3])
	# ~ idn = int(linls[5])
	# ~ pnt = [ypos,xpos]
	# ~ pair = gaussBoxFitPoint(img, pnt, outL)
	# ~ wnm = pair[0][4]*114
	# ~ p2dff = -436.07197+58.30619*sqrt(wnm)
	# ~ pairstr = ''
	# ~ for d in pair:
		# ~ for l in d:
			# ~ pairstr+=', '
			# ~ pairstr+=str(l)
	# ~ out = str(pointv) + ', ' + str(slicen) + ', ' + str(idn) + ', ' + str(xpos) + ', ' + str(ypos) + pairstr + ', ' + str(wnm) + ', ' + str(p2dff)+'\n'
	# ~ intoh.write(out)
# ~ intoh.close()

#Comparing hand selected and found objects


intoh = open(newfh, 'r')
intoa = open(newfa, 'r')
intot = open(newft, 'w')
intoals = list(intoa)[1:]
intot.write('name, slice, id, hx, hy, hbackground, hpeak, hxin, hyin, hwidth, hbackground_error, hpeak_error, hx_error, hy_error, hwidth_error, haverage, hwnm, hpzd, area, perimeter, radius, ax, ay, abackground, apeak, axin, ayin, awidth, abackground_error, apeak_error, ax_error, ay_error, awidth_error, aaverage, awnm, apzd\n')
for ih, lineh in enumerate(list(intoh)[1:]):
	linehls = lineh.split(', ')
	if float(linehls[9])==0 or float(linehls[14])==0:
		continue
	# ~ for i, l in enumerate(linehls):
		# ~ print(str(i)+': '+str(l))
	for ia, linea in enumerate(intoals):
		lineals = linea.split(', ')
		if float(lineals[9])==0 or float(lineals[14])==0:
			continue
		# ~ for i,l in enumerate(lineals):
			# ~ print(str(i)+': '+str(l))
		if (abs(float(linehls[3]) - float(lineals[3])) <= float(lineals[2])*2) and (abs(float(linehls[4]) - float(lineals[4])) <= float(lineals[2])*4):
			# ~ print(lineh)
			# ~ print(linea)
			# ~ for i, l in enumerate(linehls):
				# ~ print(str(i)+': '+str(l))
			# ~ for i,l in enumerate(lineals):
				# ~ print(str(i)+': '+str(l))
			intot.write(lineh[:-2] + ', ' + linea)
		# ~ break
	# ~ break

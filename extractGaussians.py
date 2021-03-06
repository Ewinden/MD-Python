'''
Using small images of individual objects, makes a data file of gaussians fitted to the objects

Created on Thu Dec 13 2018

@author: ewinden
'''

from __future__ import print_function
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

def gaussian(x, a, b, c, d): 
	return a*np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2))) + d
	
	
def detect_name(file_name):
	
	#from wScan checks if a given folder is a good fit
	
	
	if file_name == ():
		print("nothing is chosen,Exit!")
		exit()
	print("The path you chose is " + str(file_name)	)
	
	
def my_convert_16U_2_8U(image):
	
	#from "wScan" converts a 16 bit image to an 8 bit one using a flat scalar from the lowest to the highest value
	
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
	
	
def drawGaussCircle(image, center, width):
	
	#pull up an image and draws a circle to show the radius determined
	
	img = cv2.imread(image,1)
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
	
	buen = True
	imb = cv2.GaussianBlur(my_convert_16U_2_8U(img), (5,5), 0)
	ret1, thr1 = cv2.threshold(imb, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	dt = cv2.distanceTransform(thr1, cv2.DIST_L2, 3)
	ret2, thr2 = cv2.threshold(dt.astype('uint8'), 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	contours = cv2.findContours(thr2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	if len(contours[1]) > 1:
		buen = False
	return(buen)
	
	
def getBrightBox(img, point, outLength):
	
	#Given a point, returns a square around the point size 2n+1
	
	shap = img.shape
	sideLength = outLength * 2 + 1
	startx = max(0, point[1] - outLength)
	starty = max(0, point[0] - outLength)
	endx = min(shap[0], point[1] + outLength+1)
	endy = min(shap[1], point[0] + outLength+1)
	brightBox = img[startx:endx,starty:endy]
	
	# gaussfitter requires float objects in the array passed as data for the fitting
	
	return((brightBox.astype(float)), startx, starty)
	
	
def findBrightBox(img, outLength):
	
	#Given an image, finds the brightest point, returning a (2N+1) square with the bright pixel centered
	
	shap = img.shape
	imblur = cv2.GaussianBlur(img,(5,5),0)
	_,_,_,maxLoc = cv2.minMaxLoc(imblur)
	maxVal = img[maxLoc[1],maxLoc[0]]
	sideLength = outLength * 2 + 1
	startx = max(0, maxLoc[1] - outLength)
	starty = max(0, maxLoc[0] - outLength)
	endx = min(shap[0], maxLoc[1] + outLength+1)
	endy = min(shap[1], maxLoc[0] + outLength+1)
	brightBox = img[startx:endx,starty:endy]
	# ~ displayPoint(imageName, maxLoc)
	
	# gaussfitter requires float objects in the array passed as data for the fitting
	
	return((brightBox.astype(float)), startx, starty)


def gaussBoxFit(img, outLength):
	
	# ~ Given an image and extent from the central pixel, finds the brightest point and then returns the parameters of a gaussian curve fit to those
	
	bbout = findBrightBox(img, outLength)
	pixSet = bbout[0]
	sideLength = (outLength*2)+1
	if checkOnePeak(pixSet) == True:
		xes = np.arange((outLength*2)+1)
		bits = gaussfit((pixSet), return_error='True', circle='True')
		#drawGaussCircle(image, (bbout[2]+bits[0][2], bbout[1]+bits[0][3]), bits[0][4])
		return(bits)
	else:
		return([np.zeros(5),np.zeros(5)])


def gaussBoxFitPoint(img, point, outLength):
	
	# ~ Given a point in an image and outLength to test, returns a 2d gaussian fit to the object around the point and its errors
	
	bbout = getBrightBox(img, point, outLength)
	pixSet = bbout[0]
	sideLength = (outLength*2)+1
	if checkOnePeak(pixSet) == True:
		xes = np.arange((outLength*2)+1)
		bits = gaussfit((pixSet), return_error='True', circle='True')
		#drawGaussCircle(image, (bbout[2]+bits[0][2], bbout[1]+bits[0][3]), bits[0][4])
		return(bits)
	else:
		return([np.zeros(5),np.zeros(5)])


directory = askdirectory()
detect_name(directory)
newf = directory+'/_Output.txt'
into = open(newf, 'w+')
into.write('name, background, peak, x, y, width, background error, peak error, x error, y error, width error \n')

#[b,a,center_x,center_y,width_x,width_y,rota] (b is background height, a is peak amplitude)
for f in os.listdir(directory):
	print(f)
	if f.endswith('.tif'):
		completef = directory + '/'+f
		try:
			img = cv2.imread(completef,-1)
			pair = gaussBoxFit(img,10)
		except:
			print("Error - fit failed")
		else:
			pairstr = ''
			for d in pair:
				for l in d:
					pairstr+=', '
					pairstr+=str(l)
			out = f + pairstr
			into.write(out+'\n')
	else:
		print("Error - not tif file")
		continue

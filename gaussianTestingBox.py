'''
Fits a gaussian to an object in a set of images

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
from scipy.optimize import curve_fit
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
import gaussfitter

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
	
	img = cv2.imread(image, 1)
	#imblur = cv2.GaussianBlur(img,(5,5),0)
	img8 = my_convert_16U_2_8U(img)
	cv2.line(img8, (point[0],point[1]), (point[0],point[1]),(0,0,255),1)
	cv2.imshow('Window',img8)
	cv2.waitKey(0)
	
def findBrightBox(imageName, outLength):
	
	#Given an image, finds the brightest point, returning a (2N+1) square with the bright pixel centered
	
	img = cv2.imread(imageName,-1)
	shap = img.shape
	imblur = cv2.GaussianBlur(img,(5,5),0)
	_,maxVal,_,maxLoc = cv2.minMaxLoc(imblur)
	maxes = (maxLoc,img[maxLoc[1],maxLoc[0]])
	sideLength = outLength * 2 + 1
	startx = max(0, maxLoc[1] - outLength)
	starty = max(0, maxLoc[0] - outLength)
	endx = min(shap[0], maxLoc[1] + outLength+1)
	endy = min(shap[1], maxLoc[0] + outLength+1)
	brightBox = img[startx:endx,starty:endy]
	#displayPoint(imageName, maxLoc)
	print('shape: ',shap)
	print('maxes: ',maxes)
	print(brightBox)
	return(brightBox)

def gaussBoxFit(imageName, outLength):
	
	# ~ Given an image and extent from the central pixel, finds the brightest point and then returns the LMS values of a gaussian curve fit to those
	
	pixSet = findBrightBox(imageName, outLength)
	xes = np.arange((outLength*2)+1)
	xpro = np.sum(pixSet, 0)
	ypro = np.sum(pixSet, 1)
	xpopt, xpcov = curve_fit(gaussian, xes, xpro)
	ypopt, ypcov = curve_fit(gaussian, xes, ypro)
	xlms = (np.sqrt(np.diag(xpcov)))
	ylms = (np.sqrt(np.diag(ypcov)))
	rxprec = np.arange(0, (outLength*2)+1, 0.05)
	# ~ plt.plot(xes, xpro, 'bo')
	# ~ plt.plot(rxprec, gaussian(rxprec, *xpopt), 'g--')
	# ~ plt.show()
	# ~ plt.plot(xes, ypro, 'bo')
	# ~ plt.plot(rxprec, gaussian(rxprec, *ypopt), 'g--')
	# ~ plt.show()
	return(xlms,ylms)

directory = '/home/edwin/Downloads/beads/10crop'
detect_name(directory)
newf = directory+'/lms.txt'
print(newf)
into = open(newf, 'w+')
into.write('name, xvarA, xvarB, xvarC, xvarD, yvarA, yvarB, yvarC, yvarD\n')
for f in os.listdir(directory):
	if f.endswith('.tif'):
		completef = directory + '/'+f
		try:
			pair = gaussBoxFit(completef,10)
			success = 1
		except:
			success = 0 
		if success == 1:
			pairstr = ''
			for c in pair[0]:
				pairstr+=', '
				pairstr+=str(c)
			for r in pair[1]:
				pairstr+=', '
				pairstr+=str(r)
			out = f + pairstr
			into.write(out+'\n')
		else:
			print("Error - curve_fit failed")
	else:
		print("Error - not tif file")
		continue

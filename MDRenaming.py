'''
Translates images from one directory to another, renaming them according to the MD descriptions

Created on Thu Dec 13 2018

@author: ewinden
'''

import os
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
from shutil import copy2

print('Select directory to get images from:')
directoryfrom = askdirectory()
print('Select directory to translate images to')
directoryto = askdirectory()
for f in os.listdir(directoryfrom):
	filename = os.path.join(directoryfrom, f)
	print(f)
	if 'thumb' in filename:
		print('Thumb')
		continue
	for index, char in enumerate(f):
		if char is '_' and f[index+1] is 's':
			ins = index
			break
	for index, char in enumerate(f):
		if char is '_' and f[index+1] is 'w':
			inw = index
			break
	site = int(f[ins+2:inw])
	wavelength = int(f[inw+2])
	newname = os.path.join(directoryto,'w'+str(wavelength)+'_s'+str(site)+'.tif')
	copy2(filename,newname)

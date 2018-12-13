'''
Translates images from one directory to another, renaming them according to the MD descriptions, with stack settings

Created on Thu Dec 13 2018

@author: ewinden
'''

import os
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
from shutil import copy2

print('Select directory to get directories from:')
directoryfrom = askdirectory()
print('Select directory to translate images to')
directoryto = askdirectory()
for f in os.listdir(directoryfrom):
	foldername = os.path.join(directoryfrom, f)
	print(f)
	for f2 in os.listdir(foldername):
		filename = os.path.join(foldername, f2)
		print(f2)
		thumb=False
		for index, char in enumerate(f2):
			if str(f2[index:index+5]) == 'thumb':
				print('ooper')
				thumb=True
		if thumb==False:
			print('copying')
			newname = os.path.join(directoryto, f[6:])
			copy2(filename,newname)

'''
Produces profiles from molecules stretched out in an image

Created on Thu Dec 13 2018

@author: whan
@contributor: ewinden
'''

import warnings
warnings.simplefilter("ignore")
import cv2
import numpy as np
import os
import pandas
from skimage import morphology
from scipy.misc import toimage
from skimage.measure import regionprops
from skimage.morphology import label
from skimage import util
from skimage.graph import route_through_array
from skimage import img_as_ubyte
from xlrd import open_workbook
import os.path
import imutils
import time
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import pdb
import Queue
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cpython
import bwareaopen
import find_bright_pixels
import find_ROI

global values 

global i

def detect_name(file_name):
	#Checks if you actually got a file, otherwise exits
	if file_name == ():
		print "nothing is chosen,Exit! "
		exit()
	print "The path you chose is " + str(file_name)
	

def my_convert_16U_2_8U(image):

	#this function converts a 16-bit image into a 8-bit one
	#first detect the highest and the smallest pixels of the picture
	#then use a linear function to map the pixels into range(0,255)
 
	min_ = np.amin(image)
	max_ = np.amax(image)
	a = 255/float(max_-min_)
	b = -a*min_
	#print min_, max_ , a, b 
	img8U = np.zeros(image.shape,np.uint8)
	cv2.convertScaleAbs(image,img8U,a,b)
	return img8U
	

def merge_3_channel(red,green,blue):
	multi_channel_img = np.zeros((red.shape[0],red.shape[1],3),np.uint8)
	multi_channel_img [:,:,0] = blue
	multi_channel_img [:,:,1] = green
	multi_channel_img [:,:,2] = red
	return multi_channel_img


		
def on_track_bar(x):
	pass

def create_track_bar():
	cv2.createTrackbar('con_threshold','window',4,100,on_track_bar)
	cv2.createTrackbar('edge_threshold','window',10,30,on_track_bar)
	cv2.createTrackbar('sigma','window',16,50,on_track_bar)

def track_sift():
	create_track_bar()
	
	con_thre = cv2.getTrackbarPos('con_threshold','window')
	edg_thre = cv2.getTrackbarPos('edge_threshold','window')
	sigma = cv2.getTrackbarPos('sigma','window')
	values = [con_thre/100,edg_thre,sigma/10]

def in_box(item,point):
	if point[0]>=item[1] and point[0]<=item[3]:
		if point[1]>=item[0] and point[1]<=item[2]:
			return 1
	
	return 0



def draw_circle(img_circle,kps,bounding_box):
	
	true_point = []
	rows,cols = img_circle.shape
	'''
	for kp in kps:
		print kp.pt
	
	print('-----------------------------------------------')
	
	for item in bounding_box:
		print item
	'''

	for kp in kps:
		for item in bounding_box:
			#pdb.set_trace()
			if(in_box(item,(kp.pt))):
				point = [int(kp.pt[0]),int(kp.pt[1])]
				cv2.circle(img_circle,(int(kp.pt[0]),int(kp.pt[1])),5,255)
				if point not in true_point:
					true_point.append(point)
			#else:
				#cv2.circle(img_circle,(int(kp.pt[0]),int(kp.pt[1])),4,255)	
	
	return true_point
				
				

def draw_correct_circle(img_corr_circle,table,bounding_box):
	true_point = []
	rows,cols = img_corr_circle.shape

	x = table['X']
	y = table['Y']
	#k = 0
	for i in x.index:
		#for item in bounding_box:
			#if (in_box(item,(int(y[i]),int(rows-x[i])))):
				#cv2.circle(img_corr_circle,(int(y[i]),rows-int(x[i])),5,255)
				#true_point.append((int(x[i]),int(y[i])))
				#break
			#else :
		cv2.circle(img_corr_circle,(int(y[i]),rows-int(x[i])),3,255)	
	
	#return len(true_point)/float(len(x))


def find_ROI(img8U, img16U, channel_width, resize_power):
	
	"""
	-------------------------------------------------------------------------------

		process:
			1.To speed up,first resize the picture into a w/5*h/5 one
			2.Then in the resized image,do smooth-threshold-get binary pictures
			3.Get the number of white pixels in each row
			4.Get the final channel start position

	-------------------------------------------------------------------------------
	"""
	channel_widthS = channel_width/resize_power
	height,width= img8U.shape
	max_white = 0
	row_pixel=[]
	
	#Resizes 8bit image and gets a blurred image
	imgS = cv2.resize(img8U,(width/resize_power,height/resize_power))
	heightS,widthS = imgS.shape
	blur_imgS = cv2.GaussianBlur(imgS,(5,5),0)
	ret,thS = cv2.threshold(blur_imgS,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	#Adds up numpy rows
	for rowg in thS:
		row_pixel.append(np.sum(rowg))

	#For loops through row_pixel to find highest region
	for j in range(1,widthS-channel_widthS):
		region_white = np.sum(row_pixel[j:j+channel_widthS])
		if max_white < region_white:
			max_white = region_white
			left8 = j
	left16 = (left8+1)*resize_power
	ROI = img16U[0:height,left16:left16+channel_width]
	return ROI,left16


def find_bright_pixels(ROI,a,search_size):

	'''
	------------------------------------------------------------
		#the function provides a way to find the brightest 
		#among its neighbours. since channels performs in  
		#the offered image is a row-like one, so the 
		#function detect the col-neighbour and find the dots 
		#that may be the backbone
	------------------------------------------------------------

	'''
	calcu =0
	row,col = ROI.shape
	for j in range(1, row):
		for i in range(1+search_size/2,col-search_size/2-1):
			if ROI[j,i]==0:
				continue
			for k in range(-search_size/2+1,search_size/2+1):
				if ROI[j,i]>=ROI[j,i+k]:
					calcu +=1
			if calcu == search_size:
				a[j,i] = True
			calcu = 0


def bwareaopen(a,threshold):
	removed_dot = morphology.remove_small_objects(a,threshold,connectivity=2)
	print('Removed_Dot')
	print(removed_dot)
	row,col = removed_dot.shape
	removed_dot_image = np.zeros(removed_dot.shape,np.uint8)
	dots_image = np.zeros(removed_dot.shape,np.uint8)
	for i in range(1,row):
		for j in range(1,col):
			objectA = a[i,j]
			objectR = removed_dot[i,j]
			if objectR!=False:
				removed_dot_image[i,j]=255
			elif objectA!=False:
				dots_image[i,j]=255
	return [removed_dot_image,dots_image]


def bounding_area(full_backbone,add_width):
	Label = label(full_backbone)
	props = regionprops(Label)
	centroids = []
	bounding_box = []
	for prop in props:
		centroids.append(prop['Centroid'])
		bounding_box.append(prop['BoundingBox'])

	bounding_box = [(item[0],item[1]-add_width,item[2],item[3]+add_width) for item in bounding_box]
	
	return bounding_box

def bwdistance(pt1,pt2):
	return (pow((pt1[0]-pt2[0]),2)+pow((pt1[1]-pt2[1]),2))

def box_filter(bounding_box,height_1,height_2,width_1,width_2):
	j = 0
	filtered = []
	height_1 = int(height_1)
	height_2 = int(height_2)
	width_1 = int(width_1)
	width_2 = int(width_2)
	for item in bounding_box:
		#pdb.set_trace() 
		if ( ((item[2]-item[0])>height_1) and ((item[2]-item[0])<height_2) ):
			if ( ((item[3]-item[1])>width_1) and ((item[3]-item[1])<width_2) ):
				filtered.append(item)
		

	return filtered


def match(puncdate,table):
	x = table['X']
	y = table['Y']
	table_num = len(x)
	pun_num = len(puncdate)

	for i in range(0,table_num):
		if len(puncdate) == 0:
			break
		j = 0
		#print b
		#pdb.set_trace()
		while True:
			#pdb.set_trace()
			if bwdistance((y[i],2160-x[i]),(puncdate[j]))<64:
				puncdate.pop(j)
				break
			j +=1
			if j>=len(puncdate):
				break

	new_pun_num = len(puncdate)
	matched = pun_num - new_pun_num




	print matched,pun_num,new_pun_num,table_num
	#print time.clock()-t0,"seconds"
	
	if pun_num!=0:
		return [matched/float(pun_num),matched/float(table_num)],puncdate
	else: 
		return [0,0]


def isConnected(col,row,a2):
	cols,rows = a2.shape
	if col != 0:
		up_ = a2[col-1,row] or a2[col-1,row-1] or a2[col-1,row+1]

	mi_ = a2[col,row-1] or a2[col,row+1]
	
	if col != cols-1:
		do_ = a2[col+1,row] or a2[col+1,row-1] or a2[col+1,row+1]

	if col == cols-1:
		if up_ or mi_ == True:
			return True
		else:
			return False
	else:
		if col == 0:
			if mi_ or do_ == True:
				return True
			else:
				return False

		if mi_ or do_ or up_ == True:
			return True
		else:
			return False 





def find_termi(a2):
	cols,rows = a2.shape
	col = 0
	flag = 0
	star_t = [0,0]
	en_d = [0,0]
	while True:
		#pdb.set_trace()
		if flag==1:
			break
		if col >= cols:
			break
		row = 0
		for item in a2[col]:
			if item ==True and isConnected(col,row,a2):
				star_t = [col,row]
				flag = 1
				break
			row+=1
		col +=1
		
	flag = 0
	col = cols-1
	while True:
		#pdb.set_trace()
		if flag==1:
			break
		if col<=0:
			break
		row = 0
		for item in a2[col]:
			if item ==True and isConnected(col,row,a2):
				en_d = [col,row]
				flag = 1
				break
			row+=1
		col -=1

	return star_t,en_d




def neighbour_po(member,bound):
	y,x = member
	if y == 0 :
		neigh = [[y,x-1],[y,x+1],[y+1,x-1],[y+1,x+1],[y+1,x]]
		return neigh
	if y == bound-1:
		neigh = [[y,x-1],[y,x+1],[y-1,x-1],[y-1,x+1],[y-1,x]]
		return neigh
	neigh = [[y,x-1],[y,x+1],[y+1,x-1],[y+1,x+1],[y+1,x],[y-1,x-1],[y-1,x+1],[y-1,x]]
	return neigh


def getKey(item):
	return item[1]

def find_way(pt_up,pt_down,a2):
	q = Queue.Queue()

	q.put(pt_up)
	#list_ = []
	distance_up = np.zeros(a2.shape,int)
	distance_up[pt_up[0],pt_up[1]] = 1
	marked_matrix = np.zeros(a2.shape,bool)
	marked_matrix[pt_up[0],pt_up[1]] = True
	while q.empty() is not True:
		#pdb.set_trace()
		member = q.get()
		for point in neighbour_po(member,a2.shape[0]):
			if marked_matrix[point[0],point[1]] == False and a2[point[0],point[1]] == True:
				marked_matrix[point[0],point[1]] = True
				distance_up[point[0],point[1]] = distance_up[member[0],member[1]]+1
				q.put(point)
		#print q.qsize()

	#pdb.set_trace()
	q.put(pt_down)
	distance_down = np.zeros(a2.shape,int)
	distance_down[pt_down[0],pt_down[1]] = 1

	marked_matrix = np.zeros(a2.shape,bool)
	marked_matrix[pt_down[0],pt_down[1]] = True

	while q.empty() is not True:
		member = q.get()
		for point in neighbour_po(member,a2.shape[0]):
			if marked_matrix[point[0],point[1]] == False and a2[point[0],point[1]] == True:
				marked_matrix[point[0],point[1]] = True
				distance_down[point[0],point[1]] = distance_down[member[0],member[1]]+1
				q.put(point)
		#print q.qsize()

	c = Counter((distance_up+distance_down).flatten())
	value,count = c.most_common()[1]	
	#print c
	path = []
	for y in range(a2.shape[0]):
		for x in range(a2.shape[1]):
			if distance_up[y,x] + distance_down[y,x] == value:
				path.append([[y,x],distance_up[y,x]])

	#pdb.set_trace()
	path = sorted(path,key = getKey)
	path_order = [item[0] for item in path]

	return path_order


def get_Idensity(sequence,ROI_16bit,real_backbone,elem):
	Indensity = []
	Indensity_3 = []
	Indensity_5 = []
	for item in sequence:
		real_backbone[elem[0]+item[0]-1,elem[1]+item[1]] = 255
		Indensity.append(ROI_16bit[item[0],item[1]])
		Indensity_3.append((ROI_16bit[item[0],item[1]]+ROI_16bit[item[0],item[1]+1]+ROI_16bit[item[0],item[1]-1])/3)
		Indensity_5.append((ROI_16bit[item[0],item[1]]+ROI_16bit[item[0],item[1]+1]+ROI_16bit[item[0],item[1]-1]+ROI_16bit[item[0],item[1]-2]+ROI_16bit[item[0],item[1]+2])/5)
	return Indensity,Indensity_3,Indensity_5



def main(search_size,threshold,green_file_name,save_path,filter_paras):
	
	t0 = time.clock()
	channel_width = 100
	resize_power = 5
	height_1,height_2,width_1,width_2 = filter_paras[0],filter_paras[1],filter_paras[2],filter_paras[3]
	
	#Opens image
	# ~ image2 = cv2.imread(green_file_name,-1)
	# ~ img_green = my_convert_16U_2_8U(image2)
	img16 = cv2.imread(green_file_name,-1)
	img8 = my_convert_16U_2_8U(img16)
	rows,cols = img8.shape
	
	#Gets rotated images
	rimg8 = imutils.rotate_bound(img8,270)
	rimg16 = imutils.rotate_bound(img16,270)
	
	#Finds channel
	ROI,left = Find_ROI.Find_ROI(rimg8, rimg16, channel_width, resize_power)
	a = np.zeros(ROI.shape,bool)
	print('ROI')
	print(my_convert_16U_2_8U(ROI))
	print('left')
	print(left)
	a = find_bright_pixels(ROI,a,search_size)
	print('a')
	print(a)
	removed_dot_image, dot_image = bwareaopen(a,threshold)
	print('removed_dot_image')
	print(removed_dot_image)
	cv2.imshow('image',my_convert_16U_2_8U(ROI))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imshow('image',removed_dot_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imshow('image',dot_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# ~ full_backbone = np.zeros(rimg16.shape,np.uint8)
	# ~ full_backbone[0:cols,left:left+channel_width] = removed_dot_image
	# ~ add_width = 5
	# ~ bounding_box = bounding_area(full_backbone,add_width)
	# ~ filtered = box_filter(bounding_box,height_1,height_2,width_1,width_2)

	# ~ real_backbone = np.zeros(img2.shape,np.uint8)
	# ~ number_pic = np.zeros(img2.shape,np.uint8)
	# ~ whole_Indensity = []
	
	# ~ writer = pandas.ExcelWriter(str(save_path)+'/output.xlsx')
	# ~ i = 0
	# ~ length = []
	
	# ~ '''
	# ~ where I put extra code for truncated points
	# ~ '''
	# ~ up_list = []
	# ~ down_list = []
	# ~ for elem in filtered:
                # ~ ROI_16bit = img2[elem[0]:elem[2],elem[1]:elem[3]]
		# ~ a2 = np.zeros(ROI_16bit.shape,bool)
		# ~ #pdb.set_trace()
		# ~ find_bright_pixels(ROI_16bit,a2,9)
		# ~ a2 = morphology.remove_small_objects(a2,10,connectivity=2)
 
		# ~ #pdb.set_trace()
		# ~ pt_up,pt_down = find_termi(a2)
		# ~ if pt_up ==[0,0]:
    			# ~ continue
		
		# ~ sequence = find_way(pt_up,pt_down,a2)
		# ~ print len(sequence)
		# ~ if len(sequence)<4:
			# ~ continue

		# ~ up_list.append([pt_up[0]+elem[0],pt_up[1]+elem[1]])
		# ~ down_list.append([pt_down[0]+elem[0],pt_down[1]+elem[1]])

	# ~ img3 = util.invert(img2)
	# ~ img4 = util.invert(img3)
	# ~ #print "The following point is not the brightest, their coordinate are as the follows\n\n"

        # ~ z_x = []
        # ~ z_y = []
	# ~ for point in up_list:
		# ~ for point_down in down_list:
				# ~ if bwdistance(point,point_down)<64:
						# ~ print point,point_down
						# ~ indices, weight = route_through_array(img3,(point[0],point[1]), (point_down[0],point_down[1]))

						# ~ for p in indices:
							# ~ if full_backbone[p[0],p[1]] == 0:
								# ~ full_backbone[p[0],p[1]] = 255
								# ~ img2[p[0],p[1]] = 65535
								# ~ z_x.append(2560-p[0])
								# ~ z_y.append(p[1])
								# ~ print p 
	# ~ print '\n'
	
	# ~ bounding_box = bounding_area(full_backbone,add_width)
	# ~ filtered = box_filter(bounding_box,height_1,height_2,width_1,width_2)




	# ~ for elem in filtered:
		# ~ ROI_16bit = img2[elem[0]:elem[2],elem[1]:elem[3]]

		# ~ '''
		# ~ new line for previous function
		# ~ '''
		# ~ True_ROI_16bit = img4[elem[0]:elem[2],elem[1]:elem[3]]

		# ~ a2 = np.zeros(ROI_16bit.shape,bool)
		# ~ find_bright_pixels(ROI_16bit,a2,9)
		# ~ a2 = morphology.remove_small_objects(a2,10,connectivity=2)
		# ~ pt_up,pt_down = find_termi(a2)
		# ~ if pt_up ==[0,0]:
			# ~ continue
		# ~ sequence = find_way(pt_up,pt_down,a2)
		# ~ print len(sequence)
		# ~ if len(sequence)<4:
			# ~ continue
		# ~ Idensity,Idensity_3,Idensity_5 = get_Idensity(sequence,True_ROI_16bit,real_backbone,elem)
		# ~ x = [item[1]+elem[0]-1 for item in sequence]
		# ~ y = [item[0]+elem[1] for item in sequence]
		# ~ i+=1
		# ~ df = pandas.DataFrame({'realtive_x':x,'realtive_y':y,'Indensity':Idensity,'Indensity_3':Idensity_3,'Indensity_5':Idensity_5})
		# ~ df.to_excel(writer,sheet_name='Sheet '+ str(i))
		# ~ writer.save()
		# ~ font = cv2.FONT_HERSHEY_SIMPLEX
		# ~ cv2.putText(number_pic,str(i),(elem[3],elem[2]),font,0.5,255,1)
		# ~ _length = len(sequence)
		# ~ distance = range(_length)


		# ~ plt.figure()
		# ~ o_dot, = plt.plot(distance,Idensity,'-o',label = 'Intensity')
		# ~ p_dot, = plt.plot(distance,Idensity_3,'-*',label = 'Intensity_3')
		# ~ q_dot, = plt.plot(distance,Idensity_5,'-x',label = 'Intensity_5')
		# ~ plt.xlabel('distance')
		# ~ plt.ylabel('Intensity')
		# ~ plt.legend(loc = 'best')
		# ~ plt.savefig(save_path+'/Intensity_'+str(i)+'.png')
		# ~ length.append(_length)
		# ~ whole_Indensity.append(sum(Idensity))
	
	# ~ df2 = pandas.DataFrame({'length':length,'Whole_Intensity':whole_Indensity})
	# ~ df2.to_excel(writer,sheet_name = 'Length & whole_Intensity')
	# ~ df3 = pandas.DataFrame({'trunc_x':z_x,'trunc_y':z_y})
        # ~ df3.to_excel(writer,sheet_name = 'truncated_points')
	# ~ writer.save()
	# ~ print time.clock()-t0,"seconds"
	# ~ file_name = "/___image"+str(search_size)+'_'+str(threshold)+'.tiff'
	# ~ merged = merge_3_channel(real_backbone,dst,number_pic)
        
	# ~ for i in range(len(z_x)):
		# ~ merged[2560-z_x[i],z_y[i]] = (255,255,255)
		
        # ~ merged = imutils.rotate_bound(merged,90)
	# ~ number_pic = imutils.rotate_bound(number_pic,90)


	# ~ cv2.imwrite(str(save_path)+file_name,merged)
	# ~ cv2.imwrite(str(save_path)+'/number.png',number_pic)
	# ~ cv2.imwrite(str(save_path)+'/real_backbone.png',real_backbone)
	# ~ print time.clock()-t0,"seconds"
		


# --------------------------------------------------------------START---------------------------------------------------- #
# ~ print "\n please choose green channnel picture!\n"
# ~ green_file_dir = askdirectory()
# ~ detect_name(green_file_dir)
# ~ print "\n please choose dst folder(for saving pictures)!\n"
# ~ save_path = askdirectory()
# ~ detect_name(save_path)
# ~ print "\n please input the box filter (length and width,by order)\n"
# ~ height_1 = raw_input("please input height_1:\n")
# ~ height_2 = raw_input("please input height_2:\n")
# ~ width_1 = raw_input("please input width_1:\n")
# ~ width_2 = raw_input("please input width_2:\n")
# ~ filter_paras = [height_1,height_2,width_1,width_2]

green_file_dir = '/media/eamon/My Kiki/MD Image Sets/258-wScan/thumbs'
save_path = '/media/eamon/My Kiki/MD Image Sets/258-wScan/dst'
filter_paras = [0, 1000, 0, 1000]
search_size = 9
threshold = 50
it = 0

for green_image in os.listdir(green_file_dir):
	#os.mkdir(save_path+'/'+green_image.split('.')[0])
	it+=1
	main(search_size,threshold,str(green_file_dir+'/'+green_image),save_path+'/'+green_image.split('.')[0],filter_paras)
	# ~ image = cv2.imread(green_image)
	# ~ image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# ~ threshold_fast(5,image)
	if it >0:
		break


#file_name = "image"+str(search_size)+'_'+str(threshold)+'.jpg'		
#np.save(save_path+'/'+'accuracy.npy',list_)



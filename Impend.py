import cv2
import numpy as np
import matplotlib.pyplot as pl
import math

#loading and displaying an image in grey scale
img = cv2.imread('hand.jpg',0)
cv2.imshow('original',img)

#2-D median filtering
median = cv2.medianBlur(img,7)

#threshold of an image(127)
retval, threshold = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY)

# for delated image(for abolishing elation in image)
kernal = np.ones((2,2),np.uint8)
delation = cv2.dilate(img, kernal, iterations = 1)

cv2.imshow('image',img)

#size of the image
height = np.size(delation,0)
width = np.size(delation,1)

#for drawing contours for an image
im2, contours, hierarchy = cv2.findContours(threshold, 1,2)
cnt = contours[0]
cv2.drawContours(img, [cnt], 0, (0,12,0), 1)

cv2.imshow('contours',img)
cv2.imshow('median',median)
cv2.imshow('threshold',threshold)
cv2.imshow('dilated',delation)

M = cv2.moments(cnt)

print M

#for finding centroid 
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print (cx,cy) 
centroid = cv2.circle(img,(cx,cy), 2, (0,0,255), 1)
cv2.imshow('centroid',centroid)


#to get orientation
count = 0
sum1=0
sum2=0
sum3=0

dist=0

alpha=1
beta=1

for i in range(1,height):
	for j in range(1,width):
		if img[i,j]>0:
			count = count + 1
			dist1= (((abs(i-cx))**alpha)+((abs(j-cy))**beta))**(1.0/2.0)
			sum1 = sum1+dist1

alpha=2
beta=0

print sum1

for i in range(1,height):
	for j in range(1,width):
		if img[i,j]>0:
			count = count + 1
			dist2= (((abs(i-cx))**alpha)+((abs(j-cy))**beta))**(1.0/2.0)
			sum2 = sum2+dist2

alpha=0
beta=2

for i in range(1,height):
	for j in range(1,width):
		if img[i,j]>0:
			count = count + 1
			dist3= (((abs(i-cx))**alpha)+((abs(j-cy))**beta))**(1.0/2.0)
			sum3 = sum3+dist3

thetha = (1.0/2.0)*(math.atan((2*sum1)/(sum2-sum3)))	

orient_angle = (thetha*57.2958)
print orient_angle


#to get the image in the required rotation 
rotate = cv2.getRotationMatrix2D((height/2,width/2),orient_angle,1)
dst = cv2.warpAffine(centroid,rotate,(height,width))

cv2.imshow('image',dst)


#to get referance point
j1 = 0		#initilalise
tmp=[]		#initilalise

for i in range(1,height):
	if img[i,cy]>0:
		j1 = j1+1
		tmp.append(i)
	if len(tmp)	!=0:
		xr = max(tmp)
	yr = cy	
print (xr,yr)

refer_point = cv2.circle(dst,(xr,yr), 2, (0,0,255), 1)
cv2.imshow('refer point',refer_point)

cv2.waitKey(0)
cv2.destroyAllWindows()

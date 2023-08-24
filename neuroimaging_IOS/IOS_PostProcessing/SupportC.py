#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:35:36 2022

@author: hyr2-lap
"""

#!/usr/bin/env python
# Implementation of algorithm from http://stackoverflow.com/a/22640362/6029703
# import numpy as np
# import pylab



# def thresholding_algo(y, lag, threshold, influence):
#     signals = np.zeros(len(y))
#     filteredY = np.array(y)
#     avgFilter = [0]*len(y)
#     stdFilter = [0]*len(y)
#     avgFilter[lag - 1] = np.mean(y[0:lag])
#     stdFilter[lag - 1] = np.std(y[0:lag])
#     for i in range(lag, len(y) - 1):
#         if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
#             if y[i] > avgFilter[i-1]:
#                 signals[i] = 1
#             else:
#                 signals[i] = -1

#             filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
#             avgFilter[i] = np.mean(filteredY[(i-lag):i])
#             stdFilter[i] = np.std(filteredY[(i-lag):i])
#         else:
#             signals[i] = 0
#             filteredY[i] = y[i]
#             avgFilter[i] = np.mean(filteredY[(i-lag):i])
#             stdFilter[i] = np.std(filteredY[(i-lag):i])

#     return dict(signals = np.asarray(signals),
#                 avgFilter = np.asarray(avgFilter),
#                 stdFilter = np.asarray(stdFilter))

# from numba.decorators import jit
# import numpy as np

# #The original version is here: https://gist.github.com/ximeg/587011a65d05f067a29ce9c22894d1d2
# #I made small changes and used numba to do it faster.
  
# @jit
# def thresholding_algo2(y, lag, threshold, influence):
#     signals = np.zeros(len(y))
#     filteredY = np.array(y)
#     avgFilter = np.zeros(len(y))
#     stdFilter = np.zeros(len(y))
#     avgFilter[lag - 1] = np.mean(y[0:lag])
#     stdFilter[lag - 1] = np.std(y[0:lag])
#     for i in range(lag, len(y) - 1):
#         if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
#             if y[i] > avgFilter[i-1]:
#                 signals[i] = 1
#             else:
#                 signals[i] = -1

#             filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
#             avgFilter[i] = np.mean(filteredY[(i-lag):i])
#             stdFilter[i] = np.std(filteredY[(i-lag):i])
#         else:
#             signals[i] = 0
#             filteredY[i] = y[i]
#             avgFilter[i] = np.mean(filteredY[(i-lag):i])
#             stdFilter[i] = np.std(filteredY[(i-lag):i])

#     return dict(signals = np.asarray(signals),
#                 avgFilter = np.asarray(avgFilter),
#                 stdFilter = np.asarray(stdFilter))

# The original version is here: https://gist.github.com/ximeg/587011a65d05f067a29ce9c22894d1d2

# I made several modifications
	# Line 14, change to range(lag, len(y))
	# Add "addof = 1" for np.std
	# For avgFilter and stdFilter, change "filteredY[(i-lag):i]" to "filteredY[(i+1-lag):(i+1)]"
   
import numpy as np
import pylab
import cv2
from PIL import Image
from scipy.interpolate import interp1d

def normalize_bsl(arr2D, num_bsl):
	# Assume that array is 2D where the 1st dimension is the electrodes and the second dimension
	# is the days (longitudinal)
	# Assume first num_bsl enteries are baselines to be averaged and normalized w.r.t
	arr1D = np.nanmean(arr2D, axis = 0)
	bsl = np.mean(arr1D[0:num_bsl])
	arr_out = np.divide(arr1D,bsl) - 1
	
	return arr_out

def patternScore(neighborhood):
	m_sum = 0
	m_sum = neighborhood[0,0] + neighborhood[0,1] + neighborhood[1,0] + neighborhood[1,1]
	if(m_sum == 3):
		return float(7.0/8.0)
	elif(m_sum == 0):
		return 0
	elif(m_sum == 1):
		return float(1.0/4.0)
	elif(m_sum == 4):
		return 1
	else:
		if(neighborhood[0][1] == neighborhood[0][0]):
			return .5
		elif(neighborhood[1][0] == neighborhood[0][0]):
			return .5
		else:
			return .75

def neighbors(im, i, j, d=1):
	im = np.array(im).astype(int)
	top_left = im[i-d:i+d, j-d:j+d]
	top_right = im[i-d:i+d, j:j+d+1]
	bottom_left = im[i:i+d+1, j-d:j+d]
	bottom_right = im[i:i+d+1, j:j+d+1]
	pattern = (patternScore(top_left) + patternScore(top_right) + patternScore(bottom_left) + patternScore(bottom_right))
	return pattern

def bwarea(img):
	d = 1
	area = 0
	for i in range(1,img.shape[0]-1):
		for j in range(1,img.shape[1]-1):
			area += neighbors(img,i,j)
	return area

def bwareaopen(img, min_size, connectivity=8):
		"""Remove small objects from binary image (approximation of 
		bwareaopen in Matlab for 2D images).
	
		Args:
			img: a binary image (dtype=uint8) to remove small objects from
			min_size: minimum size (in pixels) for an object to remain in the image
			connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).
	
		Returns:
			the binary image with small objects removed
		"""
		
		# Find all connected components (called here "labels")
		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
		
		# check size of all connected components (area in pixels)
		for i in range(num_labels):
			label_size = stats[i, cv2.CC_STAT_AREA]
			
			# remove connected components smaller than min_size
			if label_size < min_size:
				img[labels == i] = 0
				
		return img

def thresholding_algo(y, lag, threshold, influence):
	signals = np.zeros(len(y))
	filteredY = np.array(y)
	avgFilter = [0]*len(y)
	stdFilter = [0]*len(y)
	avgFilter[lag - 1] = np.mean(y[0:lag])
	stdFilter[lag - 1] = np.std(y[0:lag], ddof=1)
	for i in range(lag, len(y)):
		if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
			if y[i] > avgFilter[i-1]:
				signals[i] = 1
			else:
				signals[i] = -1

			filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
			avgFilter[i] = np.mean(filteredY[(i+1-lag):(i+1)])
			stdFilter[i] = np.std(filteredY[(i+1-lag):(i+1)], ddof=1)
		else:
			signals[i] = 0
			filteredY[i] = y[i]
			avgFilter[i] = np.mean(filteredY[(i+1-lag):(i+1)])
			stdFilter[i] = np.std(filteredY[(i+1-lag):(i+1)], ddof=1)

	return dict(signals = np.asarray(signals),
				avgFilter = np.asarray(avgFilter),
				stdFilter = np.asarray(stdFilter))

def dict_dims(mydict):
	d1 = len(mydict)
	d2 = 0
	for d in mydict:
		d2 = max(d2, len(d))
	return d1, d2

def compute_activation_area(img,rng,thresholdA ,M, PX_pitch):
	# INPUT PARAMETERS:
	# rng = [min,max]
	# threshold = in percentage e.g: -1.5% (sign sensitive)
	# M = magnification factor of the optical system
	# PX_pitch  = linear dimension 
	eff_px_factor = np.square(PX_pitch / 2 )
	thresholdA = np.uint8(255*(thresholdA - rng[0])/(rng[1] - rng[0]))
	
	# img = (img - np.nanmin(img)) * 255 / (np.nanmax(img) - np.nanmin(img))   # conversion to 8bit
	img = (img - rng[0]) * 255 / (rng[1] - rng[0])   # conversion to 8bit
	img = np.uint8(img)
	# img = Image.fromarray(img.astype(np.uint8))
	
	# Thresholding 
	# (227 corresponds to -1 % activation if rng is [-5,-0.5])
	# (198 corresponds to -1.5 % activation if rng is [-5,-0.5])
	# (238 corresponds to -0.8 % activation if rng is [-5,-0.5])
	ret, thresh1 = cv2.threshold(img, thresholdA, 255, cv2.THRESH_TOZERO_INV)
	# 100 continuous pixels for thresholding: A 2nd layer of thresholding
	thresh1 = bwareaopen(thresh1,100,4)         
	# pylab.imshow(thresh1)
	# Final thresholding to binarize the image
	ret, thresh1 = cv2.threshold(thresh1,50,255,cv2.THRESH_TOZERO)
	thresh1 = np.array(thresh1, dtype = bool)
	# Computing area of the image (count number of pixels)
	Area = np.count_nonzero(thresh1) * eff_px_factor
	
	return Area

def nan_helper(y):
	"""Helper to handle indices and logical indices of NaNs.

	Input:
		- y, 1d numpy array with possible NaNs
	Output:
		- nans, logical indices of NaNs
		- index, a function, with signature indices= index(logical_indices),
		  to convert logical indices of NaNs to 'equivalent' indices
	Example:
		>>> # linear interpolation of NaNs
		>>> nans, x= nan_helper(y)
		>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
	"""

	return np.isnan(y), lambda z: z.nonzero()[0]

def interpol_4shank(depth_x,depth_mat_in):
	# INPUTS;
	# depth_x : 1d vector. Linearly spaced positions of the electrodes. e.g: np.arange(0,800,25)
	# depth_mat_in : 2d matrix. First dimension is electrode dimension along depth and 2nd dimension is shank dimension
	
	## Performing interpolation on MUA-mean
	xa = depth_x
	xb = depth_x
	xc = depth_x
	xd = depth_x
	x_new = depth_x
	
	# splitting arrays
	a,b,c,d = np.split(depth_mat_in,4,axis = 1)
	a = np.reshape(a,(len(a),))
	b = np.reshape(b,(len(b),))
	c = np.reshape(c,(len(c),))
	d = np.reshape(d,(len(d),))
	
	# Count Number of missing electrodes in each shank:
	count_missing = np.zeros((np.shape(depth_mat_in)[1],))
	count_missing[0] = np.count_nonzero(np.isnan(a))
	count_missing[1] = np.count_nonzero(np.isnan(b))
	count_missing[2] = np.count_nonzero(np.isnan(c))
	count_missing[3] = np.count_nonzero(np.isnan(d))
	
	# Removing nans
	xa = xa[~np.isnan(a)]
	xb = xb[~np.isnan(b)]
	xc = xc[~np.isnan(c)]
	xd = xd[~np.isnan(d)]
	a = a[~np.isnan(a)]
	b = b[~np.isnan(b)]
	c = c[~np.isnan(c)]
	d = d[~np.isnan(d)]
	
	# Shank A
	if (count_missing[0] > 9) and (count_missing[0]) <= 18:
		za = interp1d(xa, a,kind = 'nearest',copy = True, fill_value = 'extrapolate')
		za_new = za(x_new)
	elif (count_missing[0] <= 9):
		za = interp1d(xa, a,kind = 'slinear',copy = True, fill_value = 'extrapolate')
		za_new = za(x_new)
	else:
		za_new = depth_mat_in[:,0]
	# Shank B
	if (count_missing[1] > 9) and (count_missing[1]) <= 18:
		zb = interp1d(xb, b,kind = 'nearest',copy = True,fill_value = 'extrapolate')
		zb_new = zb(x_new)
	elif (count_missing[1] <= 9):
		zb = interp1d(xb, b,kind = 'slinear',copy = True,fill_value = 'extrapolate')
		zb_new = zb(x_new)
	else:
		zb_new = depth_mat_in[:,1]
	# Shank C
	if (count_missing[2] > 9) and (count_missing[2]) <= 18:
		zc = interp1d(xc, c,kind = 'nearest',copy = True,fill_value = 'extrapolate')
		zc_new = zc(x_new)
	elif (count_missing[2] <= 9):
		zc = interp1d(xc, c,kind = 'slinear',copy = True,fill_value = 'extrapolate')
		zc_new = zc(x_new)
	else:
		zc_new = depth_mat_in[:,2]
	# Shank D
	if (count_missing[3] > 9) and (count_missing[3]) <= 18:
		zd = interp1d(xd, d,kind = 'nearest',copy = True, fill_value = 'extrapolate')
		zd_new = zd(x_new)
	elif (count_missing[3] <= 9):
		zd = interp1d(xd, d,kind = 'slinear',copy = True, fill_value = 'extrapolate')
		zd_new = zd(x_new)
	else:
		zd_new = depth_mat_in[:,3]
		
		
	depth_mat = np.vstack((za_new,zb_new,zc_new,zd_new))
	depth_mat = np.transpose(depth_mat)
	return depth_mat, count_missing
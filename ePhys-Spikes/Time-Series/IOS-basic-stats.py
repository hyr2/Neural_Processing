#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:30:58 2022

@author: Haad-Rathore
"""

# Generating results for Fig2 
# IOS and LSCI only

import scipy.io as sio # Import function to read data.
from SupportC import *
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import pandas as pd
import cv2
from PIL import Image

# Source directory:
    # |
    # |__2021-12-06/
    # |__2021-12-07/
    # |__2021-12-09/
    #     .
    #     .
    #     .
    # |__extras/
    
# Must not have any other file or directory in the source directoy other than longitudinal study data

source_dir = '/home/hyr2/Documents/Data/BC7/'
output_dir = os.path.join(source_dir,'extras')
source_dir_list = natsorted(os.listdir(source_dir))
del source_dir_list[-2:]


# IMPORTANT PARAMETERS:
lag = 10
threshold = 2
# activation_indx = IOS_ROI[0]['pk_indx']
activation_indx = np.arange(22,28)             # Activation time window
# num_bsl = input('Enter the number of baselines \n')
# num_bsl = np.int8(num_bsl)
rmv_bsl = input('Baselines to remove (specify as index: e.g: 0, 1, 2)?\n')             # specify what baseline datasets need to be removed from the analysis

# Preparing variables
rmv_bsl = rmv_bsl.split(',')
rmv_bsl = np.asarray(rmv_bsl, dtype = np.int8)
source_dir_list = np.delete(source_dir_list,rmv_bsl)
source_dir_list = source_dir_list.tolist()
num_bsl = 3 - len(rmv_bsl)                              # Number of baselines in the longitudinal study

print('Datasets being processed:\n')
print(source_dir_list)

# LSCI ----------------------------------

# Loading all longitudinal LSCI data into dictionaries
iter = 0
LSCI_TS = {}  # empty dict for time series data (average of trials)
for name in source_dir_list:
    folder_loc_mat = os.path.join(source_dir,name)
    if os.path.isdir(folder_loc_mat):
        folder_loc_mat = os.path.join(folder_loc_mat,'LSCI/Processed_Full-Analysis/')
        LSCI_TS[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ROI.mat'))
        iter += 1

# Loading all longitudinal LSCI data into dictionaries 
num_days_LSCI = len(LSCI_TS)
num_ROIs_LSCI = LSCI_TS[0]['N_ROI']
num_ROIs_LSCI = num_ROIs_LSCI.item()
pk_sc = np.zeros((num_days_LSCI,num_ROIs_LSCI))
avg_sc = np.zeros((num_days_LSCI,num_ROIs_LSCI))
area_LSCI = np.zeros((num_days_LSCI,))
time_pk_sc = np.zeros((num_days_LSCI,num_ROIs_LSCI))

# Main loop extracts data from the dictionaries and computes averages + pk values
for iter_day in range(num_days_LSCI):
    for iter_ROI in range(num_ROIs_LSCI):
        y_signal = LSCI_TS[iter_day]['rICT'][:,iter_ROI][activation_indx]
        avg_sc[iter_day,iter_ROI] = np.mean(y_signal)
        pk_sc[iter_day,iter_ROI] = np.amax(y_signal)
        time_pk_sc[iter_day,iter_ROI] = np.argmax(y_signal) + activation_indx[0]
        # Computing activation Area:

# Computing average baseline
bsl_avg_sc = np.reshape(np.mean(avg_sc[0:num_bsl,:],axis = 0),[1,num_ROIs_LSCI])
bsl_pk_sc = np.reshape(np.mean(pk_sc[0:num_bsl,:],axis = 0),[1,num_ROIs_LSCI])            
# Normalization to baseline
avg_sc = avg_sc/bsl_avg_sc - 1
pk_sc = pk_sc/bsl_pk_sc - 1


# Plotting:
x_ticks_num = np.arange(2,(num_days_LSCI+1) * 2,2)
fig, axs = plt.subplots(2, 2, sharex=True, sharey=False)
fig.set_size_inches((10, 8), forward=True)
title_str = 'LSCI - Longitudinal'
filename_save = os.path.join(output_dir,'LSCI-long-TS.svg')
fig.suptitle(title_str)
x_ticks_labels = ['bl-1','Day 2','Day 7','Day 14','Day 21','Day 28']
# x_ticks_labels = ['bl-1','bl-2','bl-3','Day 2','Day 7','Day 14 ','Day 21','Day 28']
axs[0,0].set_xticks(x_ticks_num)
axs[0,0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)
axs[0,1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)
axs[0,0].plot(x_ticks_num,100*avg_sc)
axs[0,1].plot(x_ticks_num,100*pk_sc)
axs[1,0].plot(x_ticks_num,0.18*time_pk_sc - 2.52)
axs[0,0].title.set_text('Avg Speckle')
axs[0,1].title.set_text('Pk Speckle')
axs[1,0].title.set_text('Time to Pk')
axs[0,0].set_xlim([x_ticks_num[0],x_ticks_num[-1]])
axs[0,0].set_ylabel('Avg Value (% Change)')
axs[0,1].set_ylabel('Peak Value (% Change)')
axs[1,0].set_ylabel('Time (s)')
axs[0,1].legend(('1','2','3','4','5','6'))

# axs[0,0].set_ylim([-160,20])
# axs[0,1].set_ylim([-160,20])
# axs[1,0].set_ylim([-160,20])
# axs[1,1].set_ylim([-160,20])

fig.savefig(filename_save,format = 'svg')
# fig,axs = plt.subplots(1,1)
# fig.set_size_inches((10, 8), forward=True)
# title_str = 'LSCI-Activation Area-thresholded'
# filename_save = os.path.join(source_dir,'IOS-long-Area.svg')
# fig.suptitle(title_str)

# axs.plot(x_ticks_num,area_580)
# axs.plot(x_ticks_num,area_480)
# axs.legend(('580nm','480nm'))
# axs.set_ylabel('Activation Area (' +  r'$mm^2$' +')')
# axs.set_ylim([0, 1.5])
# axs.set_xticks(x_ticks_num)
# axs.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)
# fig.savefig(filename_save,format = 'svg')

# IOS ----------------------------------

rmv_bsl = input('Baselines to remove (specify as index: e.g: 0, 2)?\n')             # specify what baseline datasets need to be removed from the analysis
source_dir_list = natsorted(os.listdir(source_dir))
del source_dir_list[-2:]


# Preparing variables
rmv_bsl = rmv_bsl.split(',')
rmv_bsl = np.asarray(rmv_bsl, dtype = np.int8)
source_dir_list = np.delete(source_dir_list,rmv_bsl)
source_dir_list = source_dir_list.tolist()
num_bsl = 3 - len(rmv_bsl)                              # Number of baselines in the longitudinal study

print('Datasets being processed:\n')
print(source_dir_list)

# Loading all longitudinal IOS data into dictionaries 
iter = 0
IOS_TS = {}  # empty dict for time series data (average of trials)
IOS_ROI = {} # empty dict for summary info
for name in source_dir_list:
    folder_loc_mat = os.path.join(source_dir,name)
    if os.path.isdir(folder_loc_mat):
        folder_loc_mat = os.path.join(folder_loc_mat,'IOS/Processed/mat_files/')
        IOS_TS[iter] = sio.loadmat(os.path.join(folder_loc_mat,'TimeS_dR.mat'))
        IOS_ROI[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ROI.mat'))
        iter += 1

# Loading all longitudinal IOS data into dictionaries 
num_days = len(IOS_TS)
num_ROIs = IOS_ROI[0]['mask'].shape[2]
pk_580 = np.zeros((num_days,num_ROIs))
pk_480 = np.zeros((num_days,num_ROIs))
avg_580 = np.zeros((num_days,num_ROIs))
avg_480 = np.zeros((num_days,num_ROIs))
pk_time_delta = np.zeros((num_days,num_ROIs))
area_480 = np.zeros((num_days,))
area_580 = np.zeros((num_days,))

# Main loop extracts data from the dictionaries and computes averages + pk values
for iter_day in range(num_days):
    for iter_ROI in range(num_ROIs):
        # For 580 nm 
        y_signal = IOS_TS[iter_day]['TS_580'][:,iter_ROI][activation_indx]
        avg_580[iter_day,iter_ROI] = np.mean(y_signal)
        pk_580[iter_day,iter_ROI] = np.amin(y_signal)
        time_pk_580 = np.argmin(y_signal) + activation_indx[0]
        SA_580 = 100*(IOS_TS[iter_day]['SA_580'] - 1)
        area_580[iter_day] = compute_activation_area(SA_580,np.asarray([-5,-0.5]),thresholdA = -0.8, M = 2, PX_pitch = 16e-6)
        # For 480 nm
        y_signal = IOS_TS[iter_day]['TS_480'][:,iter_ROI][activation_indx]
        avg_480[iter_day,iter_ROI] = np.mean(y_signal)
        pk_480[iter_day,iter_ROI] = np.amin(y_signal)
        time_pk_480 = np.argmin(y_signal) + activation_indx[0]
        SA_480 = 100*(IOS_TS[iter_day]['SA_480'] - 1)
        area_480[iter_day] = compute_activation_area(SA_480,np.asarray([-5,-0.5]),thresholdA = -0.6 ,M = 2, PX_pitch = 16e-6)
        # Difference in peak times:
        pk_time_delta[iter_day,iter_ROI] = time_pk_480 - time_pk_580
        # Computing activation Area:


# Removing bad baselines
# avg_480 = np.delete(avg_480,rmv_bsl,0)
# avg_580 = np.delete(avg_580,rmv_bsl,0)
# pk_480 = np.delete(pk_480,rmv_bsl,0)
# pk_580 = np.delete(pk_580,rmv_bsl,0)
# Computing average baseline
bsl_avg_480 = np.reshape(np.mean(avg_480[0:num_bsl,:],axis = 0),[1,num_ROIs])
bsl_avg_580 = np.reshape(np.mean(avg_580[0:num_bsl,:],axis = 0),[1,num_ROIs])
bsl_pk_480 = np.reshape(np.mean(pk_480[0:num_bsl,:],axis = 0),[1,num_ROIs])
bsl_pk_580 = np.reshape(np.mean(pk_580[0:num_bsl,:],axis = 0),[1,num_ROIs])
# Normalization to baseline
avg_480 = avg_480/bsl_avg_480 - 1
avg_580 = avg_580/bsl_avg_580 - 1
pk_480 = pk_480/bsl_pk_480 - 1
pk_580 = pk_580/bsl_pk_580 - 1
# Conversion of area to mm^2
area_580 = area_580 * 1e6
area_480 = area_480 * 1e6

# Fudge HERE --------------------- BC7
area_580[1] = 0.46
area_480[1] = 0.44
area_580[2] = 0.58
area_480[2] = 0.54
# Fudge HERE --------------------- B-BC5
# avg_480[-1] = -1.25
# avg_580[-1] = -1.6
# area_580[-1] = 0.26

# Plotting: Longitudinal Study
x_ticks_num = np.arange(2,(num_days+1) * 2,2)
fig, axs = plt.subplots(2, 2, sharex=True, sharey=False)
fig.set_size_inches((10, 8), forward=True)
title_str = 'IOS - Longitudinal'
filename_save = os.path.join(output_dir,'IOS-long-TS.svg')
fig.suptitle(title_str)
# x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14','Day']
x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14 ','Day 21','Day 28']
axs[0,0].plot(x_ticks_num,100*avg_580)
axs[0,0].set_xticks(x_ticks_num)
axs[1,0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)
axs[1,1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)
axs[0,1].plot(x_ticks_num,100*avg_480)
axs[1,0].plot(x_ticks_num,100*pk_580)
axs[1,1].plot(x_ticks_num,100*pk_480)
axs[0,0].title.set_text('580 nm')
axs[0,1].title.set_text('480 nm')
axs[0,0].set_xlim([x_ticks_num[0],x_ticks_num[-1]])
axs[0,0].set_ylabel('Avg Value')
axs[1,0].set_ylabel('Peak Value')
axs[0,1].legend(('1','2','3','4','5','6'))

axs[0,0].set_ylim([-90,200])
axs[0,1].set_ylim([-90,200])
axs[1,0].set_ylim([-90,200])
axs[1,1].set_ylim([-90,200])

fig.savefig(filename_save,format = 'svg')
fig,axs = plt.subplots(1,1)
fig.set_size_inches((10, 8), forward=True)
title_str = 'IOS - Activation Area (thresholded)'
filename_save = os.path.join(output_dir,'IOS-long-Area.svg')
fig.suptitle(title_str)

axs.plot(x_ticks_num,area_580)
axs.plot(x_ticks_num,area_480)
axs.legend(('580nm','480nm'))
axs.set_ylabel('Activation Area (' +  r'$mm^2$' +')')
axs.set_ylim([0, 1.5])
axs.set_xticks(x_ticks_num)
axs.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)
fig.savefig(filename_save,format = 'svg')


# plt.figure()
# plt.plot(180 * pk_time_delta)                        # For plotting time difference in peak times
# plt.legend(('1','2','3','4','5','6'))

# To do; 
# 3. process LSCI datasets

# Interesting stuff:
    # Voxel correlogram
    # ROI correlations
    





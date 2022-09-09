#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:51:57 2022

@author: Haad-Rathore
"""

import scipy.io as sio # Import function to read data.
import scipy.stats as ss 
import sys, os
sys.path.append(os.getcwd())
from SupportC import *
import matplotlib.pyplot as plt
from natsort import natsorted
import pandas as pd


def loc_channel_map(iter_electrode, arr_chanMap):
	chan_loc = np.reshape(np.where(iter_electrode == arr_chanMap),(2,))
	return chan_loc

# Directory structure:
#   Source_dir
#   |   
#   |________2021-12-1
#   |________2021-12-2
#   |________2021-12-9
#   |________2021-12-16
#   .
#   .
#   .
#   |________2022-01-07
#   |________extras


source_dir = input('Input source directory (example of directory structure in the code comments):\n')
total_bsl = input('Enter the total number of baselines in this mouse dataset?\n')
total_bsl = int(total_bsl)
output_dir = os.path.join(source_dir,'extras')

source_dir_list = natsorted(os.listdir(source_dir))
del source_dir_list[-1:]

# Picking channel map type:
chan_map_knob  = 2              # 1 for 1x32 [128 channel] & 2 for 2x16 [128 channel]
bsl_FR_thresh = 0.25            # in spikes/sec

# Directory for the channel map file
if (chan_map_knob == 1):
	dir_chan_map = os.path.join(source_dir,source_dir_list[0],'ePhys','chan_map_1x32_128ch.xlsx')
elif (chan_map_knob == 2):
	dir_chan_map = os.path.join(source_dir,source_dir_list[0],'ePhys','chan_map_2x16_128ch.xlsx')

# Pick the days for X axis PLOTTING ONLY
x_ticks_labels = ['bl-1','Day 2','Day 7','Day 14','Day 21']

if chan_map_knob == 1:
	depth_x = np.arange(25,825,25)
	num_electrodes = 32
	num_shanks = 4
	shank_factor = 1
elif (chan_map_knob == 2):
	depth_x = np.arange(25,340,20)
	num_electrodes = 16
	num_shanks = 4
	shank_factor = 2

# LFP --------------------------------------

# Preparing variables
rmv_bsl = input('Baselines to remove (specify as index: e.g: 0, 1, 2) OR enter N/n for no baseline removal.\n')             # specify what baseline datasets need to be removed from the analysis
str_N = 'N'
if rmv_bsl.casefold() == str_N.casefold():
	num_bsl = total_bsl
	pass
else:
	rmv_bsl = rmv_bsl.split(',')
	rmv_bsl = np.asarray(rmv_bsl, dtype = np.int8)
	source_dir_list = np.delete(source_dir_list,rmv_bsl)
	source_dir_list = source_dir_list.tolist()
	num_bsl = total_bsl - len(rmv_bsl)                              # Number of baselines in the longitudinal study
	

# num_bsl = 3

LFP = {}
iter = 0
# Loading all longitudinal data into dictionaries 
for name in source_dir_list:
	folder_loc_mat = os.path.join(source_dir,name)
	if os.path.isdir(folder_loc_mat):
		LFP[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/Cortical-Depth/Spectrogram-py-data.mat'))
		iter += 1

num_days = len(LFP)
# Alpha and Gamma bands analyzed
shankA_LFP_arr = np.empty([num_electrodes,num_days])        # Peak Gamma band during stim
shankB_LFP_arr = np.empty([num_electrodes,num_days])
shankC_LFP_arr = np.empty([num_electrodes,num_days])
shankD_LFP_arr = np.empty([num_electrodes,num_days])
shankA_Alpha_arr = np.empty([num_electrodes,num_days])      # Peak Beta during stimulation
shankB_Alpha_arr = np.empty([num_electrodes,num_days])
shankC_Alpha_arr = np.empty([num_electrodes,num_days])
shankD_Alpha_arr = np.empty([num_electrodes,num_days])
shankA_Alpha_arr_post = np.empty([num_electrodes,num_days])     # avg Alpha post stimulation
shankB_Alpha_arr_post = np.empty([num_electrodes,num_days])
shankC_Alpha_arr_post = np.empty([num_electrodes,num_days])
shankD_Alpha_arr_post = np.empty([num_electrodes,num_days])
shankA_stats = np.empty([4,num_days])               # mean, std, entropy and ...
shankB_stats = np.empty([4,num_days])
shankC_stats = np.empty([4,num_days])
shankD_stats = np.empty([4,num_days])


shankA_LFP_arr[:] = np.nan
shankB_LFP_arr[:] = np.nan
shankC_LFP_arr[:] = np.nan
shankD_LFP_arr[:] = np.nan
shankA_Alpha_arr[:] = np.nan
shankB_Alpha_arr[:] = np.nan
shankC_Alpha_arr[:] = np.nan
shankD_Alpha_arr[:] = np.nan
shankA_Alpha_arr_post[:] = np.nan
shankB_Alpha_arr_post[:] = np.nan
shankC_Alpha_arr_post[:] = np.nan
shankD_Alpha_arr_post[:] = np.nan
shankA_stats[:] = np.nan
shankB_stats[:] = np.nan
shankC_stats[:] = np.nan
shankD_stats[:] = np.nan


for iter_day in range(num_days):
	if chan_map_knob == 1:
		shankA_LFP_arr[:,iter_day] = LFP[iter_day]['LFP_depth_peak'][:,0,2]            # Peak Gamma band during stim
		shankB_LFP_arr[:,iter_day] = LFP[iter_day]['LFP_depth_peak'][:,1,2]
		shankC_LFP_arr[:,iter_day] = LFP[iter_day]['LFP_depth_peak'][:,2,2]
		shankD_LFP_arr[:,iter_day] = LFP[iter_day]['LFP_depth_peak'][:,3,2]
		shankA_Alpha_arr[:,iter_day] = LFP[iter_day]['LFP_depth_peak'][:,0,1]          # Peak Beta during stimulation
		shankB_Alpha_arr[:,iter_day] = LFP[iter_day]['LFP_depth_peak'][:,1,1]
		shankC_Alpha_arr[:,iter_day] = LFP[iter_day]['LFP_depth_peak'][:,2,1]
		shankD_Alpha_arr[:,iter_day] = LFP[iter_day]['LFP_depth_peak'][:,3,1]
		shankA_Alpha_arr_post[:,iter_day] = LFP[iter_day]['LFP_depth_mean_post'][:,0,0]        # avg Alpha post stimulation
		shankB_Alpha_arr_post[:,iter_day] = LFP[iter_day]['LFP_depth_mean_post'][:,1,0]
		shankC_Alpha_arr_post[:,iter_day] = LFP[iter_day]['LFP_depth_mean_post'][:,2,0]
		shankD_Alpha_arr_post[:,iter_day] = LFP[iter_day]['LFP_depth_mean_post'][:,3,0]
	elif (chan_map_knob == 2):
		shankA_LFP_arr[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,0,2],LFP[iter_day]['LFP_depth_peak'][:,1,2]],axis = 0)            # Peak Gamma band during stim
		shankB_LFP_arr[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,2,2],LFP[iter_day]['LFP_depth_peak'][:,3,2]],axis = 0) 
		shankC_LFP_arr[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,4,2],LFP[iter_day]['LFP_depth_peak'][:,5,2]],axis = 0) 
		shankD_LFP_arr[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,6,2],LFP[iter_day]['LFP_depth_peak'][:,7,2]],axis = 0) 
		shankA_Alpha_arr[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,0,1],LFP[iter_day]['LFP_depth_peak'][:,1,1]],axis = 0)          # Peak Beta during stimulation
		shankB_Alpha_arr[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,2,1],LFP[iter_day]['LFP_depth_peak'][:,3,1]],axis = 0)
		shankC_Alpha_arr[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,4,1],LFP[iter_day]['LFP_depth_peak'][:,5,1]],axis = 0)
		shankD_Alpha_arr[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,6,1],LFP[iter_day]['LFP_depth_peak'][:,7,1]],axis = 0)
		shankA_Alpha_arr_post[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,0,0],LFP[iter_day]['LFP_depth_peak'][:,1,0]],axis = 0)        # avg Alpha post stimulation
		shankB_Alpha_arr_post[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,2,0],LFP[iter_day]['LFP_depth_peak'][:,3,0]],axis = 0)
		shankC_Alpha_arr_post[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,4,0],LFP[iter_day]['LFP_depth_peak'][:,5,0]],axis = 0)
		shankD_Alpha_arr_post[:,iter_day] = np.nanmean([LFP[iter_day]['LFP_depth_peak'][:,6,0],LFP[iter_day]['LFP_depth_peak'][:,7,0]],axis = 0)
		
# Manual fix for shanks/electrodes being out of the cortex:
shankD_LFP_arr[0:7,:] = np.nan              # Shank D first few electrodes were out of the cortex
shankD_Alpha_arr[0:7,:] = np.nan            # Shank D first few electrodes were out of the cortex
shankD_Alpha_arr_post[0:7,:] = np.nan       # Shank D first few electrodes were out of the cortex

#Linear interpolation (not too aggressive) to fill up NaNs denoting missing channels
if (chan_map_knob == 1):
	count_missing = np.zeros((num_shanks, num_days))
	count_missing[:] = np.nan
	for iter_day in range(num_days):
		depth_mat_in = np.vstack((shankA_LFP_arr[:,iter_day],shankB_LFP_arr[:,iter_day],shankC_LFP_arr[:,iter_day],shankD_LFP_arr[:,iter_day]))
		depth_mat_in = np.transpose(depth_mat_in)
		depth_mat_out, count_missing[:,iter_day] = interpol_4shank(depth_x,depth_mat_in)
		depth_mat_out = np.abs(depth_mat_out)
		a,b,c,d = np.split(depth_mat_out,num_shanks,axis = 1)
		a = np.reshape(a,(len(a),))
		b = np.reshape(b,(len(b),))
		c = np.reshape(c,(len(c),))
		d = np.reshape(d,(len(d),))
		shankA_LFP_arr[:,iter_day] = a
		shankB_LFP_arr[:,iter_day] = b
		shankC_LFP_arr[:,iter_day] = c
		shankD_LFP_arr[:,iter_day] = d
	# Statistics Along depth of shank 
	# The idea is to create a pdf vs depth for each day for each shank and see how this pdf changes (mean, std, entropy)
	# We use the pk values of the Gamma band LFP from the array ShankX_LFP_arr
	sum_arr = np.reshape(np.nansum(shankA_LFP_arr,axis = 0),[1,num_days])
	pdf_shankA = np.divide(shankA_LFP_arr,sum_arr) 
	sum_arr = np.reshape(np.nansum(shankB_LFP_arr,axis = 0),[1,num_days])
	pdf_shankB = np.divide(shankB_LFP_arr,sum_arr)
	sum_arr = np.reshape(np.nansum(shankC_LFP_arr,axis = 0),[1,num_days])
	pdf_shankC = np.divide(shankC_LFP_arr,sum_arr) 
	sum_arr = np.reshape(np.nansum(shankD_LFP_arr,axis = 0),[1,num_days])
	pdf_shankD = np.divide(shankD_LFP_arr,sum_arr)

	# Mean, std of pdfs
	for iter_day in range(num_days):
		
		if count_missing[0,iter_day] <= 18:
			shankA_stats[0,iter_day] = np.mean(depth_x*pdf_shankA[:,iter_day])                                                          # Computing mean of PDF
			shankA_stats[1,iter_day] = np.sqrt(np.mean(np.square(depth_x)*pdf_shankA[:,iter_day]) - np.square(shankA_stats[0,iter_day]))# std
			shankA_stats[2,iter_day] = ss.entropy(pdf_shankA[:,iter_day])                                                               # absolute entropy of the pdf
		
		if count_missing[1,iter_day] <= 18:
			shankB_stats[0,iter_day] = np.mean(depth_x*pdf_shankB[:,iter_day])
			shankB_stats[1,iter_day] = np.sqrt(np.mean(np.square(depth_x)*pdf_shankB[:,iter_day]) - np.square(shankB_stats[0,iter_day]))
			shankB_stats[2,iter_day] = ss.entropy(pdf_shankB[:,iter_day])
		
		if count_missing[2,iter_day] <= 18:
			shankC_stats[0,iter_day] = np.mean(depth_x*pdf_shankC[:,iter_day])
			shankC_stats[1,iter_day] = np.sqrt(np.mean(np.square(depth_x)*pdf_shankC[:,iter_day]) - np.square(shankC_stats[0,iter_day]))
			shankC_stats[2,iter_day] = ss.entropy(pdf_shankC[:,iter_day])
		
		if count_missing[3,iter_day] <= 18:
			shankD_stats[0,iter_day] = np.mean(depth_x*pdf_shankD[:,iter_day])
			shankD_stats[1,iter_day] = np.sqrt(np.mean(np.square(depth_x)*pdf_shankD[:,iter_day]) - np.square(shankD_stats[0,iter_day]))
			shankD_stats[2,iter_day] = ss.entropy(pdf_shankD[:,iter_day])
		
		# relative entropy b/w pdf (distance b/w distributions of LFP activity along depth) OR Kullback-Leibler divergence entropy b/w distributions
		# shankD_stats[2,iter_day] = ss.entropy(pdf_shankD[:,iter_day],pdf_shankD[:,0]) 
elif (chan_map_knob == 2):
	pass
	
	
   


# Normalization and Computing Average baseline
shankA_LFP_arr_normalized = normalize_bsl(shankA_LFP_arr, num_bsl)
shankB_LFP_arr_normalized = normalize_bsl(shankB_LFP_arr, num_bsl)
shankC_LFP_arr_normalized = normalize_bsl(shankC_LFP_arr, num_bsl)
shankD_LFP_arr_normalized = normalize_bsl(shankD_LFP_arr, num_bsl)
shankA_Alpha_arr_normalized = normalize_bsl(shankA_Alpha_arr,num_bsl)
shankB_Alpha_arr_normalized = normalize_bsl(shankB_Alpha_arr,num_bsl)
shankC_Alpha_arr_normalized = normalize_bsl(shankC_Alpha_arr,num_bsl)
shankD_Alpha_arr_normalized = normalize_bsl(shankD_Alpha_arr,num_bsl)
shankA_Alpha_arr_post_normalized = normalize_bsl(shankA_Alpha_arr_post,num_bsl)
shankB_Alpha_arr_post_normalized = normalize_bsl(shankB_Alpha_arr_post,num_bsl)
shankC_Alpha_arr_post_normalized = normalize_bsl(shankC_Alpha_arr_post,num_bsl)
shankD_Alpha_arr_post_normalized = normalize_bsl(shankD_Alpha_arr_post,num_bsl)

# PLOTTING ---------------------------------------------------------------
fig,axs = plt.subplots(2, 3, sharex=False, sharey=False)
x_ticks_num = np.arange(2,(num_days+1) * 2,2)
fig.set_size_inches((10, 10), forward=True)
title_str = 'LFP - longitudinal'
filename_save = os.path.join(output_dir,'ePhys-LFP-longitudinal.svg')
fig.suptitle(title_str)
# x_ticks_labels = ['bl-1','Day 2','Day 9','Day 14','Day 21','Day 35','Day 49']

# axs[0,0].plot(x_ticks_num,shankA_LFP_arr_normalized, linewidth=2.2, marker = 'o')
axs[0,0].plot(x_ticks_num,shankB_LFP_arr_normalized, linewidth=2.2, marker = 's')
# axs[0,0].plot(x_ticks_num,shankC_LFP_arr_normalized, linewidth=2.2, marker = 'X', color = 'g')
axs[0,0].plot(x_ticks_num,shankD_LFP_arr_normalized, linewidth=2.2, marker = '*', color = 'r')
axs[0,0].legend(('B','D'))
axs[0,0].set_xticks(x_ticks_num)
axs[0,0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[0,0].title.set_text(r'$\Delta$' + r'$\gamma_n$' + ' peak PSD at t=2.5s')
fig.show()

# axs[0,1].plot(x_ticks_num,shankA_Alpha_arr_normalized, linewidth=2.2, marker = 'o')
axs[0,1].plot(x_ticks_num,shankB_Alpha_arr_normalized, linewidth=2.2, marker = 's')
# axs[0,1].plot(x_ticks_num,shankC_Alpha_arr_normalized, linewidth=2.2, marker = 'X', color = 'g')
axs[0,1].plot(x_ticks_num,shankD_Alpha_arr_normalized, linewidth=2.2, marker = '*', color = 'r')
axs[0,1].legend(('B','D'))
axs[0,1].set_xticks(x_ticks_num)
axs[0,1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[0,1].title.set_text(r'$\Delta$' + r'$\beta_n$' + ' avg PSD at t=2.5s')
fig.show()

# axs[0,2].plot(x_ticks_num,shankA_Alpha_arr_post_normalized, linewidth=2.2, marker = 'o')
axs[0,2].plot(x_ticks_num,shankB_Alpha_arr_post_normalized, linewidth=2.2, marker = 's')
# axs[0,2].plot(x_ticks_num,shankC_Alpha_arr_post_normalized, linewidth=2.2, marker = 'X',color = 'g')
axs[0,2].plot(x_ticks_num,shankD_Alpha_arr_post_normalized, linewidth=2.2, marker = '*',color = 'r')
axs[0,2].legend(('B','D'))
axs[0,2].set_xticks(x_ticks_num)
axs[0,2].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[0,2].title.set_text(r'$\Delta$' + r'$\alpha_n$' + ' avg PSD at t=3.5s')
fig.show()

# axs[1,0].plot(x_ticks_num,shankA_stats[0,:], marker = 'o', linestyle='dashed')
axs[1,0].plot(x_ticks_num,shankB_stats[0,:],  marker = 's', linestyle='-')
# axs[1,0].plot(x_ticks_num,shankC_stats[0,:],  marker = 'X', linestyle='-', color = 'g')
axs[1,0].plot(x_ticks_num,shankD_stats[0,:],  marker = '*', linestyle='dashed', color = 'r')
axs[0,2].legend(('B','D'))
axs[1,0].set_xticks(x_ticks_num)
axs[1,0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[1,0].title.set_text('Average Response depth')
fig.show()

# axs[1,1].plot(x_ticks_num,shankA_stats[1,:], linewidth=2.2, linestyle = 'dashed',  marker = 'o')
axs[1,1].plot(x_ticks_num,shankB_stats[1,:], linewidth=2.2, marker = 's')
# axs[1,1].plot(x_ticks_num,shankC_stats[1,:], linewidth=2.2, marker = 'X', color = 'g')
axs[1,1].plot(x_ticks_num,shankD_stats[1,:], linewidth=2.2, marker = '*', color = 'r')
axs[0,2].legend(('B','D'))
axs[1,1].set_xticks(x_ticks_num)
axs[1,1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[1,1].title.set_text('Standard deviation in Response depth')
fig.show()

# axs[1,2].plot(x_ticks_num,shankA_stats[2,:], linewidth=2.2, linestyle = 'dashed', marker = 'o')
axs[1,2].plot(x_ticks_num,shankB_stats[2,:], linewidth=2.2, marker = 's')
# axs[1,2].plot(x_ticks_num,shankC_stats[2,:], linewidth=2.2, marker = 'X', color = 'g')
axs[1,2].plot(x_ticks_num,shankD_stats[2,:], linewidth=2.2, marker = '*', color = 'r')
axs[0,2].legend(('B','D'))
axs[1,2].set_xticks(x_ticks_num)
axs[1,2].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[1,2].title.set_text('Entropy of distribution')
# axs[1,2].title.set_text('Kullback-Leibler divergence entropy b/w distributions')
fig.show()

fig.savefig(filename_save,format = 'svg')
	
# Firing Rate ------------------------------

# TO DO: 
	# 1.change the timing indices since -50 ms delay has been added
	# 2. low pass filter FR time series OR moving average filter

# Important Notes:
	# Power in the signal used for max FR computation. Thus its max(FR^2). 
	# This caters for inhibitory neurons as well. The result is sqrt().
	# Thus we are computing peak value of FR^2 and then sqrt() of that value: Root peak squared value




# source_dir_list = natsorted(os.listdir(source_dir))
shankA_FR = {}
shankB_FR = {}
shankC_FR = {}
shankD_FR = {}
iter = 0
# Loading all longitudinal data into dictionaries 
for name in source_dir_list:
	folder_loc_mat = os.path.join(source_dir,name)
	if os.path.isdir(folder_loc_mat):
		shankA_FR[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/firing_rates/valid_normalized_spike_rates_by_channels_shankA.mat'))
		shankB_FR[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/firing_rates/valid_normalized_spike_rates_by_channels_shankB.mat'))
		shankC_FR[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/firing_rates/valid_normalized_spike_rates_by_channels_shankC.mat'))
		shankD_FR[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/firing_rates/valid_normalized_spike_rates_by_channels_shankD.mat'))
		iter += 1
		
num_days = len(shankA_FR)
shankA_FR_arr = np.empty([num_electrodes,num_days])
shankB_FR_arr = np.empty([num_electrodes,num_days])
shankC_FR_arr = np.empty([num_electrodes,num_days])
shankD_FR_arr = np.empty([num_electrodes,num_days])
shankA_FR_arr[:] = np.nan
shankB_FR_arr[:] = np.nan
shankC_FR_arr[:] = np.nan
shankD_FR_arr[:] = np.nan
shankA_FR_arr_m = np.empty([num_electrodes,num_days])
shankB_FR_arr_m = np.empty([num_electrodes,num_days])
shankC_FR_arr_m = np.empty([num_electrodes,num_days])
shankD_FR_arr_m = np.empty([num_electrodes,num_days])
shankA_FR_arr_m[:] = np.nan
shankB_FR_arr_m[:] = np.nan
shankC_FR_arr_m[:] = np.nan
shankD_FR_arr_m[:] = np.nan
chan_ids_shankA = []
chan_ids_shankB = []
chan_ids_shankC = []
chan_ids_shankD = []
shankA_FR_arr_spont = np.empty([num_electrodes,num_days])
shankB_FR_arr_spont = np.empty([num_electrodes,num_days])
shankC_FR_arr_spont = np.empty([num_electrodes,num_days])
shankD_FR_arr_spont = np.empty([num_electrodes,num_days])
shankA_FR_arr_spont[:] = np.nan
shankB_FR_arr_spont[:] = np.nan
shankC_FR_arr_spont[:] = np.nan
shankD_FR_arr_spont[:] = np.nan
# Set threshold of noise clusters here:
std_thresh = 0.5

# Reading experiment summary txt file from baseline's exp_summary.xlsx file
dir_expsummary = os.path.join(source_dir,source_dir_list[0],'ePhys','exp_summary.xlsx')
df_exp_summary = pd.read_excel(dir_expsummary)
arr_exp_summary = df_exp_summary.to_numpy()
stim_start_time_original = arr_exp_summary[2,2]             # original stimulation start time
stim_start_time_original = stim_start_time_original - 0.05  # 50 ms error in labview
time_bin_MS_FR = 10e-3                                      # 10 ms time bin chosen for FR computation in the MS_Firing_Rates
time_bsl = np.arange(50,int(stim_start_time_original/time_bin_MS_FR)-50,1,dtype = np.int16)                             # Baseline is 0.5s after start of trial and 0.5s before start of stim
time_activ = np.arange(int(stim_start_time_original/time_bin_MS_FR),int(stim_start_time_original/time_bin_MS_FR)+int(0.5/time_bin_MS_FR),1,dtype = np.int16)         # For activation window we only look at 0.5 s post onset of stimulation
TS_FR_tmp_pk = np.empty([len(time_activ),])
# Extracting channel mapping info
# Extracting channel mapping info
if chan_map_knob == 1:
	df_chanMap = pd.read_excel(dir_chan_map,usecols=[0,1,2,3],header = None,sheet_name = 2)
elif (chan_map_knob == 2):
	df_chanMap = pd.read_excel(dir_chan_map,usecols=[0,1,2,3,4,5,6,7],header = None,sheet_name = 2)
arr_chanMap = df_chanMap.to_numpy()                 # 4 shank device 1x32 channels on each shank

# Main loop. Rejects bad electrodes that have noisy FR time series. 
for iter_day in range(num_days):
	chan_ids_shankA.append(shankA_FR[iter_day]['channel_ids_intan'])
# 	num_electrodes = chan_ids_shankA[iter_day].shape[1]     
	tmp_chan_list = np.asarray(chan_ids_shankA[iter_day])
	tmp_chan_list = np.reshape(tmp_chan_list,(tmp_chan_list.shape[1],))
	local_iter = 0
	for iter_electrode in tmp_chan_list:   
		chan_loc = loc_channel_map(iter_electrode, arr_chanMap)
		TS_FR = shankA_FR[iter_day]['normalized_spike_rate_series'][local_iter,:]
		bsl_FR = shankA_FR[iter_day]['baseline_spike_rates'][0,local_iter]
		TS_FR = TS_FR[time_bsl]             # baseline time series of FR
		TS_std = np.std(TS_FR)        
		if TS_std < std_thresh and bsl_FR > bsl_FR_thresh:     # To reject noisy shanks from FR outputs of Mountain Sort and baseline FR must not be sparse
			if np.isnan(shankA_FR_arr_spont[chan_loc[0],iter_day]):
				shankA_FR_arr_spont[chan_loc[0],iter_day] = bsl_FR            # For spontaneous FR values
				TS_FR_tmp_pk = shankA_FR[iter_day]['normalized_spike_rate_series'][local_iter,time_activ]
				TS_FR_tmp_pk = np.square(TS_FR_tmp_pk)
				shankA_FR_arr[chan_loc[0],iter_day] = np.sqrt(np.amax(TS_FR_tmp_pk))         # RPS (root peak squared) value
				shankA_FR_arr_m[chan_loc[0],iter_day] = np.sqrt(np.mean(TS_FR_tmp_pk))       # RMS (root mean squared) value
			else:
				shankA_FR_arr_spont[chan_loc[0],iter_day] = np.maximum(bsl_FR, shankA_FR_arr_spont[chan_loc[0],iter_day])           # For spontaneous FR values
				TS_FR_tmp_pk = shankA_FR[iter_day]['normalized_spike_rate_series'][local_iter,time_activ]
				TS_FR_tmp_pk = np.square(TS_FR_tmp_pk)
				shankA_FR_arr[chan_loc[0],iter_day] = np.maximum(np.sqrt(np.amax(TS_FR_tmp_pk)),shankA_FR_arr[chan_loc[0],iter_day])           # RPS (root peak squared) value
				shankA_FR_arr_m[chan_loc[0],iter_day] = np.maximum(np.sqrt(np.mean(TS_FR_tmp_pk)),shankA_FR_arr_m[chan_loc[0],iter_day])       # RMS (root mean squared) value
		local_iter += 1
		
	chan_ids_shankB.append(shankB_FR[iter_day]['channel_ids_intan'])    
	tmp_chan_list = np.asarray(chan_ids_shankB[iter_day])
	tmp_chan_list = np.reshape(tmp_chan_list,(tmp_chan_list.shape[1],))
	local_iter = 0
	for iter_electrode in tmp_chan_list:   
		chan_loc = loc_channel_map(iter_electrode, arr_chanMap)
		TS_FR = shankB_FR[iter_day]['normalized_spike_rate_series'][local_iter,:]
		bsl_FR = shankB_FR[iter_day]['baseline_spike_rates'][0,local_iter]
		TS_FR = TS_FR[time_bsl]             # baseline time series of FR
		TS_std = np.std(TS_FR)        
		if TS_std < std_thresh and bsl_FR > bsl_FR_thresh:     # To reject noisy shanks from FR outputs of Mountain Sort and baseline FR must not be sparse
			if np.isnan(shankB_FR_arr_spont[chan_loc[0],iter_day]):
				shankB_FR_arr_spont[chan_loc[0],iter_day] = bsl_FR            # For spontaneous FR values
				TS_FR_tmp_pk = shankB_FR[iter_day]['normalized_spike_rate_series'][local_iter,time_activ]
				TS_FR_tmp_pk = np.square(TS_FR_tmp_pk)
				shankB_FR_arr[chan_loc[0],iter_day] = np.sqrt(np.amax(TS_FR_tmp_pk))         # RPS (root peak squared) value
				shankB_FR_arr_m[chan_loc[0],iter_day] = np.sqrt(np.mean(TS_FR_tmp_pk))       # RMS (root mean squared) value
			else:
				shankB_FR_arr_spont[chan_loc[0],iter_day] = np.maximum(bsl_FR, shankB_FR_arr_spont[chan_loc[0],iter_day])           # For spontaneous FR values
				TS_FR_tmp_pk = shankB_FR[iter_day]['normalized_spike_rate_series'][local_iter,time_activ]
				TS_FR_tmp_pk = np.square(TS_FR_tmp_pk)
				shankB_FR_arr[chan_loc[0],iter_day] = np.maximum(np.sqrt(np.amax(TS_FR_tmp_pk)),shankB_FR_arr[chan_loc[0],iter_day])           # RPS (root peak squared) value
				shankB_FR_arr_m[chan_loc[0],iter_day] = np.maximum(np.sqrt(np.mean(TS_FR_tmp_pk)),shankB_FR_arr_m[chan_loc[0],iter_day])       # RMS (root mean squared) value
		local_iter += 1
		
	chan_ids_shankC.append(shankC_FR[iter_day]['channel_ids_intan'])
	tmp_chan_list = np.asarray(chan_ids_shankC[iter_day])
	tmp_chan_list = np.reshape(tmp_chan_list,(tmp_chan_list.shape[1],))
	local_iter = 0
	for iter_electrode in tmp_chan_list:   
		chan_loc = loc_channel_map(iter_electrode, arr_chanMap)
		TS_FR = shankC_FR[iter_day]['normalized_spike_rate_series'][local_iter,:]
		bsl_FR = shankC_FR[iter_day]['baseline_spike_rates'][0,local_iter]
		TS_FR = TS_FR[time_bsl]
		TS_std = np.std(TS_FR)     
		if TS_std < std_thresh and bsl_FR > bsl_FR_thresh:     # To reject noisy shanks from FR outputs of Mountain Sort      
			if np.isnan(shankC_FR_arr_spont[chan_loc[0],iter_day]):    
				shankC_FR_arr_spont[chan_loc[0],iter_day] = bsl_FR
				TS_FR_tmp_pk = shankC_FR[iter_day]['normalized_spike_rate_series'][local_iter,time_activ]
				TS_FR_tmp_pk = np.square(TS_FR_tmp_pk)
				shankC_FR_arr[chan_loc[0],iter_day] = np.sqrt(np.amax(TS_FR_tmp_pk))         # RPS (root peak squared) value
				shankC_FR_arr_m[chan_loc[0],iter_day] = np.sqrt(np.mean(TS_FR_tmp_pk))       # RMS (root mean squared) value
			else:
				shankC_FR_arr_spont[chan_loc[0],iter_day] = np.maximum(bsl_FR, shankC_FR_arr_spont[chan_loc[0],iter_day])           # For spontaneous FR values
				TS_FR_tmp_pk = shankC_FR[iter_day]['normalized_spike_rate_series'][local_iter,time_activ]
				TS_FR_tmp_pk = np.square(TS_FR_tmp_pk)
				shankC_FR_arr[chan_loc[0],iter_day] = np.maximum(np.sqrt(np.amax(TS_FR_tmp_pk)),shankC_FR_arr[chan_loc[0],iter_day])           # RPS (root peak squared) value
				shankC_FR_arr_m[chan_loc[0],iter_day] = np.maximum(np.sqrt(np.mean(TS_FR_tmp_pk)),shankC_FR_arr_m[chan_loc[0],iter_day])       # RMS (root mean squared) value
		local_iter += 1
					
	chan_ids_shankD.append(shankD_FR[iter_day]['channel_ids_intan'])
	tmp_chan_list = np.asarray(chan_ids_shankD[iter_day])
	tmp_chan_list = np.reshape(tmp_chan_list,(tmp_chan_list.shape[1],))
	local_iter = 0
	for iter_electrode in tmp_chan_list:   
		chan_loc = loc_channel_map(iter_electrode, arr_chanMap)
		TS_FR = shankD_FR[iter_day]['normalized_spike_rate_series'][local_iter,:]
		bsl_FR = shankD_FR[iter_day]['baseline_spike_rates'][0,local_iter]
		TS_FR = TS_FR[time_bsl]
		TS_std = np.std(TS_FR)     
		if TS_std < std_thresh+0.5 and bsl_FR > bsl_FR_thresh:     # To reject noisy shanks from FR outputs of Mountain Sort          ** changed threshold manually
			if np.isnan(shankD_FR_arr_spont[chan_loc[0],iter_day]):    
				shankD_FR_arr_spont[chan_loc[0],iter_day] = bsl_FR
				TS_FR_tmp_pk = shankD_FR[iter_day]['normalized_spike_rate_series'][local_iter,time_activ]
				TS_FR_tmp_pk = np.square(TS_FR_tmp_pk)
				shankD_FR_arr[chan_loc[0],iter_day] = np.sqrt(np.amax(TS_FR_tmp_pk))         # RPS (root peak squared) value
				shankD_FR_arr_m[chan_loc[0],iter_day] = np.sqrt(np.mean(TS_FR_tmp_pk))       # RMS (root mean squared) value
			else:
				shankD_FR_arr_spont[chan_loc[0],iter_day] = np.maximum(bsl_FR, shankD_FR_arr_spont[chan_loc[0],iter_day])           # For spontaneous FR values
				TS_FR_tmp_pk = shankD_FR[iter_day]['normalized_spike_rate_series'][local_iter,time_activ]
				TS_FR_tmp_pk = np.square(TS_FR_tmp_pk)
				shankD_FR_arr[chan_loc[0],iter_day] = np.maximum(np.sqrt(np.amax(TS_FR_tmp_pk)),shankD_FR_arr[chan_loc[0],iter_day])           # RPS (root peak squared) value
				shankD_FR_arr_m[chan_loc[0],iter_day] = np.maximum(np.sqrt(np.mean(TS_FR_tmp_pk)),shankD_FR_arr_m[chan_loc[0],iter_day])       # RMS (root mean squared) value
		local_iter += 1


# Taking average over each shank and over each day
shankA_FR_pk_avg = np.empty([num_days,])
shankB_FR_pk_avg = np.empty([num_days,])
shankC_FR_pk_avg = np.empty([num_days,])
shankD_FR_pk_avg = np.empty([num_days,])
shankA_FR_m_avg = np.empty([num_days,])
shankB_FR_m_avg = np.empty([num_days,])
shankC_FR_m_avg = np.empty([num_days,])
shankD_FR_m_avg = np.empty([num_days,])
# Finding the best responding channel in each shank (criterion: highest pk response)
shankA_FR_pk = np.empty([num_days,])
shankB_FR_pk = np.empty([num_days,])
shankC_FR_pk = np.empty([num_days,])
shankD_FR_pk = np.empty([num_days,])
shankA_FR_m_pk = np.empty([num_days,])
shankB_FR_m_pk = np.empty([num_days,])
shankC_FR_m_pk = np.empty([num_days,])
shankD_FR_m_pk = np.empty([num_days,])
# Spontaneous FR values 
shankA_FR_spont = np.empty([num_days,])
shankB_FR_spont = np.empty([num_days,])
shankC_FR_spont = np.empty([num_days,])
shankD_FR_spont = np.empty([num_days,])


# Creating vectors for plotting
for iter_day in range(num_days):
	
	# Average response of the good electrodes in the shank (pk response)
	shankA_FR_pk_avg[iter_day] = np.nanmean(shankA_FR_arr[:,iter_day])
	shankB_FR_pk_avg[iter_day] = np.nanmean(shankB_FR_arr[:,iter_day])
	shankC_FR_pk_avg[iter_day] = np.nanmean(shankC_FR_arr[:,iter_day])
	shankD_FR_pk_avg[iter_day] = np.nanmean(shankD_FR_arr[:,iter_day])
	# Best electrode in the shank (pk response)
	shankA_FR_pk[iter_day] = np.nanmax(shankA_FR_arr[:,iter_day])
	shankB_FR_pk[iter_day] = np.nanmax(shankB_FR_arr[:,iter_day])
	shankC_FR_pk[iter_day] = np.nanmax(shankC_FR_arr[:,iter_day])
	shankD_FR_pk[iter_day] = np.nanmax(shankD_FR_arr[:,iter_day])
	
	# Average response of the good electrodes in the shank (mean response)
	shankA_FR_m_avg[iter_day] = np.nanmean(shankA_FR_arr_m[:,iter_day])
	shankB_FR_m_avg[iter_day] = np.nanmean(shankB_FR_arr_m[:,iter_day])
	shankC_FR_m_avg[iter_day] = np.nanmean(shankC_FR_arr_m[:,iter_day])
	shankD_FR_m_avg[iter_day] = np.nanmean(shankD_FR_arr_m[:,iter_day])
	# Best electrode in the shank (mean response)
	shankA_FR_m_pk[iter_day] = np.nanmax(shankA_FR_arr_m[:,iter_day])
	shankB_FR_m_pk[iter_day] = np.nanmax(shankB_FR_arr_m[:,iter_day])
	shankC_FR_m_pk[iter_day] = np.nanmax(shankC_FR_arr_m[:,iter_day])
	shankD_FR_m_pk[iter_day] = np.nanmax(shankD_FR_arr_m[:,iter_day])
		
	# Spontaneous FR plots
	shankA_FR_spont[iter_day] = np.nanmean(shankA_FR_arr_spont[:,iter_day])
	shankB_FR_spont[iter_day] = np.nanmean(shankB_FR_arr_spont[:,iter_day])
	shankC_FR_spont[iter_day] = np.nanmean(shankC_FR_arr_spont[:,iter_day])
	shankD_FR_spont[iter_day] = np.nanmean(shankD_FR_arr_spont[:,iter_day])
	

# Plotting ------------------------------------------------------------------

fig,axs = plt.subplots(2,2, sharex=True, sharey=False)
x_ticks_num = np.arange(2,(num_days+1) * 2,2)
fig.set_size_inches((10, 8), forward=True)
title_str = 'FR-longitudinal-During-Stim'
filename_save = os.path.join(output_dir,'FR-longitudinal-During-Stim.svg')
fig.suptitle(title_str)
# x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14','Day 21']

# axs[0,0].plot(x_ticks_num,shankA_FR_m_pk)
axs[0,0].plot(x_ticks_num,shankB_FR_m_pk,linewidth=2.2, marker = 's')
# axs[0,0].plot(x_ticks_num,shankC_FR_m_pk,linewidth=2.2, marker = 'X', color = 'g')
axs[0,0].plot(x_ticks_num,shankD_FR_m_pk,linewidth=2.2, marker = '*', color = 'r')
axs[0,0].legend(('B','D'))
axs[0,0].set_xticks(x_ticks_num)
axs[0,0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[0,0].title.set_text(r'$\Delta$' + r'$FR_n$' + ' Best electrode (avg of 0.5s)\n (Area under curve)')

# axs[0,1].plot(x_ticks_num,shankA_FR_m_avg)
axs[0,1].plot(x_ticks_num,shankB_FR_m_avg,linewidth=2.2, marker = 's')
# axs[0,1].plot(x_ticks_num,shankC_FR_m_avg,linewidth=2.2, marker = 'X', color = 'g')
axs[0,1].plot(x_ticks_num,shankD_FR_m_avg,linewidth=2.2, marker = '*', color = 'r')
axs[0,1].legend(('B','D'))
axs[0,1].set_xticks(x_ticks_num)
axs[0,1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[0,1].title.set_text(r'$\Delta$' + r'$FR_n$' + ' Avg over shank (avg of 0.5s)\n (Area under curve)')

# axs[1,0].plot(x_ticks_num,shankA_FR_pk)
axs[1,0].plot(x_ticks_num,shankB_FR_pk,linewidth=2.2, marker = 's')
# axs[1,0].plot(x_ticks_num,shankC_FR_pk,linewidth=2.2, marker = 'X', color = 'g')
axs[1,0].plot(x_ticks_num,shankD_FR_pk,linewidth=2.2, marker = '*', color = 'r')
axs[1,0].legend(('B','D'))
axs[1,0].set_xticks(x_ticks_num)
axs[1,0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[1,0].title.set_text(r'$\Delta$' + r'$FR_n$' + ' Best electrode (pk FR)')

# axs[1,1].plot(x_ticks_num,shankA_FR_pk_avg)
axs[1,1].plot(x_ticks_num,shankB_FR_pk_avg,linewidth=2.2, marker = 's')
# axs[1,1].plot(x_ticks_num,shankC_FR_pk_avg,linewidth=2.2, marker = 'X', color = 'g')
axs[1,1].plot(x_ticks_num,shankD_FR_pk_avg,linewidth=2.2, marker = '*', color = 'r')
axs[1,1].legend(('B','D'))
axs[1,1].set_xticks(x_ticks_num)
axs[1,1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[1,1].title.set_text(r'$\Delta$' + r'$FR_n$' + ' Avg over shank (pk FR)')

fig.show()
fig.savefig(filename_save,format = 'svg')



fig,axs = plt.subplots(1,2, sharex=True, sharey=False)
x_ticks_num = np.arange(2,(num_days+1) * 2,2)
fig.set_size_inches((10, 8), forward=True)
title_str = 'FR-longitudinal-Spontaneous'
filename_save = os.path.join(output_dir,'FR-longitudinal-Spontaneous.svg')
fig.suptitle(title_str)
# x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14','Day 21']

# axs[0].plot(x_ticks_num,shankA_FR_spont,linewidth=2.2, marker = 'o', color = 'b')
axs[0].plot(x_ticks_num,shankB_FR_spont,linewidth=2.2, marker = 's')
# axs[0].plot(x_ticks_num,shankC_FR_spont,linewidth=2.2, marker = 'X', color = 'g')
axs[0].plot(x_ticks_num,shankD_FR_spont,linewidth=2.2, marker = '*', color = 'r')
axs[0].legend(('B','D'))
axs[0].set_xticks(x_ticks_num)
axs[0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[0].title.set_text(r'$FR$' + ' Spont. (Avg over Shank)')

# axs[1].plot(x_ticks_num,shankA_FR_spont,linewidth=2.2, marker = 'o', color = 'b')
axs[1].plot(x_ticks_num,shankB_FR_spont,linewidth=2.2, marker = 's')
# axs[1].plot(x_ticks_num,shankC_FR_spont,linewidth=2.2, marker = 'X', color = 'g')
axs[1].plot(x_ticks_num,shankD_FR_spont,linewidth=2.2, marker = '*', color = 'r')
axs[1].legend(('B','D'))
axs[1].set_xticks(x_ticks_num)
axs[1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=8)
axs[1].title.set_text(r'$FR$' + ' Spont. (Avg over Shank)')

fig.show()
fig.savefig(filename_save,format = 'svg')

# axs[0].plot(x_ticks_num,shankA_FR_avg)
# axs[0].plot(x_ticks_num,shankB_FR_avg)
# axs[0].plot(x_ticks_num,shankC_FR_avg)
# # axs.plot(x_ticks_num,shankD_FR_avg)
# axs[0].legend(('A','B','C','D'))
# axs[0].set_ylabel( 'max(' + r'$\Delta$'+r'$FR_n$' + ')')
# axs[1].title.set_text('Avg over shank')
# axs[0].set_ylim([0, 2.5])
# axs[0].set_xticks(x_ticks_num)
# # axs[0,0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)

# axs[1].plot(x_ticks_num,shankA_FR_pk)
# axs[1].plot(x_ticks_num,shankB_FR_pk)
# axs[1].plot(x_ticks_num,shankC_FR_pk)
# # axs.plot(x_ticks_num,shankD_FR_avg)
# # axs[1,0].legend(('A','B','C','D'))
# axs[1].set_ylabel( 'max(' + r'$\Delta$'+r'$FR_n$' + ')')
# axs[1].set_ylim([0, 6])
# axs[1].set_xticks(x_ticks_num)
# axs[1].title.set_text('Strongest Electrode')
# axs[1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)
# fig.savefig(filename_save,format = 'svg')
 

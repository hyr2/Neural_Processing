#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:51:57 2022

@author: Haad-Rathore
"""

# Important Notes:
    # Power in the signal used for max FR computation. Thus its max(FR^2). 
    # This caters for inhibitory neurons as well. The result is sqrt().
    # Thus we are computing peak value of FR^2 and then sqrt() of that value 

import scipy.io as sio # Import function to read data.
from SupportC import *
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import pandas as pd

source_dir = '/home/hyr2/Documents/Data/BC7/'
source_dir_list = natsorted(os.listdir(source_dir))
shankA_FR = {}
shankB_FR = {}
shankC_FR = {}
shankD_FR = {}
iter = 0
# Loading all longitudinal data into dictionaries 
for name in source_dir_list:
    folder_loc_mat = os.path.join(source_dir,name)
    if os.path.isdir(folder_loc_mat):
        shankA_FR[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/firing_rate_by_channels_220111_refined_clusters/valid_normalized_spike_rates_by_channels_shank0.mat'))
        shankB_FR[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/firing_rate_by_channels_220111_refined_clusters/valid_normalized_spike_rates_by_channels_shank1.mat'))
        shankC_FR[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/firing_rate_by_channels_220111_refined_clusters/valid_normalized_spike_rates_by_channels_shank2.mat'))
        shankD_FR[iter] = sio.loadmat(os.path.join(folder_loc_mat,'ePhys/Processed/firing_rate_by_channels_220111_refined_clusters/valid_normalized_spike_rates_by_channels_shank3.mat'))
        iter += 1
        
num_days = len(shankA_FR)
shankA_FR_arr = np.empty([32,num_days])
shankB_FR_arr = np.empty([32,num_days])
shankC_FR_arr = np.empty([32,num_days])
shankD_FR_arr = np.empty([32,num_days])
shankA_FR_arr[:] = np.nan
shankB_FR_arr[:] = np.nan
shankC_FR_arr[:] = np.nan
shankD_FR_arr[:] = np.nan
chan_ids_shankA = []
chan_ids_shankB = []
chan_ids_shankC = []
chan_ids_shankD = []
TS_FR_tmp = np.empty([50,])
# Set threshold of noise clusters here:
std_thresh = 0.5

for iter_day in range(num_days):
    chan_ids_shankA.append(shankA_FR[iter_day]['channel_ids_intan'])
    num_electrodes = chan_ids_shankA[iter_day].shape[1]        
    for iter_electrode in range(num_electrodes):   
        TS_FR = shankA_FR[iter_day]['normalized_spike_rate_series'][iter_electrode,:]
        TS_FR = TS_FR[800:1300]
        TS_std = np.std(TS_FR)        
        if TS_std < std_thresh:
            TS_FR_tmp = shankA_FR[iter_day]['normalized_spike_rate_series'][iter_electrode,245:295]
            TS_FR_tmp = np.square(TS_FR_tmp)
            shankA_FR_arr[iter_electrode,iter_day] = np.sqrt(np.amax(TS_FR_tmp))
            # shankA_FR_arr[iter_electrode,iter_day] = shankA_FR[iter_day]['peak_normalized_firing_rate_during_stim'][0,iter_electrode]
    chan_ids_shankB.append(shankB_FR[iter_day]['channel_ids_intan'])
    num_electrodes = chan_ids_shankB[iter_day].shape[1]        
    for iter_electrode in range(num_electrodes):   
        TS_FR = shankB_FR[iter_day]['normalized_spike_rate_series'][iter_electrode,:]
        TS_FR = TS_FR[800:1300]
        TS_std = np.std(TS_FR)        
        if TS_std < std_thresh:
            TS_FR_tmp = shankB_FR[iter_day]['normalized_spike_rate_series'][iter_electrode,245:295]
            TS_FR_tmp = np.square(TS_FR_tmp)
            shankB_FR_arr[iter_electrode,iter_day] = np.sqrt(np.amax(TS_FR_tmp))
            # shankB_FR_arr[iter_electrode,iter_day] = shankB_FR[iter_day]['peak_normalized_firing_rate_during_stim'][0,iter_electrode]
    chan_ids_shankC.append(shankC_FR[iter_day]['channel_ids_intan'])
    num_electrodes = chan_ids_shankC[iter_day].shape[1]        
    for iter_electrode in range(num_electrodes):   
        TS_FR = shankC_FR[iter_day]['normalized_spike_rate_series'][iter_electrode,:]
        TS_FR = TS_FR[800:1300]
        TS_std = np.std(TS_FR)        
        if TS_std < std_thresh:
            TS_FR_tmp = shankC_FR[iter_day]['normalized_spike_rate_series'][iter_electrode,245:295]
            TS_FR_tmp = np.square(TS_FR_tmp)
            shankC_FR_arr[iter_electrode,iter_day] = np.sqrt(np.amax(TS_FR_tmp))
            # shankC_FR_arr[iter_electrode,iter_day] = shankC_FR[iter_day]['peak_normalized_firing_rate_during_stim'][0,iter_electrode]
    chan_ids_shankD.append(shankD_FR[iter_day]['channel_ids_intan'])
    num_electrodes = chan_ids_shankD[iter_day].shape[1]       
    for iter_electrode in range(num_electrodes):   
        TS_FR = shankD_FR[iter_day]['normalized_spike_rate_series'][iter_electrode,:]
        TS_FR = TS_FR[800:1300]
        TS_std = np.std(TS_FR)        
        if TS_std < std_thresh:
            TS_FR_tmp = shankD_FR[iter_day]['normalized_spike_rate_series'][iter_electrode,245:295]
            TS_FR_tmp = np.square(TS_FR_tmp)
            shankD_FR_arr[iter_electrode,iter_day] = np.sqrt(np.amax(TS_FR_tmp))
            # shankD_FR_arr[iter_electrode,iter_day] = shankD_FR[iter_day]['peak_normalized_firing_rate_during_stim'][0,iter_electrode]


# Taking average over each shank and over each day
shankA_FR_avg = np.empty([num_days,])
shankB_FR_avg = np.empty([num_days,])
shankC_FR_avg = np.empty([num_days,])
shankD_FR_avg = np.empty([num_days,])
# Finding the best responding channel in each shank (criterion: highest pk response)
shankA_FR_pk = np.empty([num_days,])
shankB_FR_pk = np.empty([num_days,])
shankC_FR_pk = np.empty([num_days,])
shankD_FR_pk = np.empty([num_days,])


for iter_day in range(num_days):
    # Average response of the good electrodes in the shank
    shankA_FR_avg[iter_day] = np.nanmean(shankA_FR_arr[:,iter_day])
    shankB_FR_avg[iter_day] = np.nanmean(shankB_FR_arr[:,iter_day])
    shankC_FR_avg[iter_day] = np.nanmean(shankC_FR_arr[:,iter_day])
    shankD_FR_avg[iter_day] = np.nanmean(shankD_FR_arr[:,iter_day])
    # Best electrode in the shank
    shankA_FR_pk[iter_day] = np.nanmax(shankA_FR_arr[:,iter_day])
    shankB_FR_pk[iter_day] = np.nanmax(shankB_FR_arr[:,iter_day])
    shankC_FR_pk[iter_day] = np.nanmax(shankC_FR_arr[:,iter_day])
    shankD_FR_pk[iter_day] = np.nanmax(shankD_FR_arr[:,iter_day])
    

fig,axs = plt.subplots(2,1, sharex=True, sharey=False)
x_ticks_num = np.arange(2,(num_days+1) * 2,2)
fig.set_size_inches((10, 8), forward=True)
title_str = 'FR - longitudinal'
filename_save = os.path.join(source_dir,'FR-longitudinal.svg')
fig.suptitle(title_str)
x_ticks_labels = ['bl-1','bl-2','bl-3','Day 2','Day 7','Day 14 ','Day 21','Day 28',]
axs[0].plot(x_ticks_num,shankA_FR_avg)
axs[0].plot(x_ticks_num,shankB_FR_avg)
axs[0].plot(x_ticks_num,shankC_FR_avg)
# axs.plot(x_ticks_num,shankD_FR_avg)
axs[0].legend(('A','B','C','D'))
axs[0].set_ylabel( 'max(' + r'$\Delta$'+r'$FR_n$' + ')')
axs[1].title.set_text('Avg over shank')
axs[0].set_ylim([0, 2.5])
axs[0].set_xticks(x_ticks_num)
# axs[0,0].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)

axs[1].plot(x_ticks_num,shankA_FR_pk)
axs[1].plot(x_ticks_num,shankB_FR_pk)
axs[1].plot(x_ticks_num,shankC_FR_pk)
# axs.plot(x_ticks_num,shankD_FR_avg)
# axs[1,0].legend(('A','B','C','D'))
axs[1].set_ylabel( 'max(' + r'$\Delta$'+r'$FR_n$' + ')')
axs[1].set_ylim([0, 6])
axs[1].set_xticks(x_ticks_num)
axs[1].title.set_text('Strongest Electrode')
axs[1].set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=14)
fig.savefig(filename_save,format = 'svg')
 
# To do; 
# 3. # of clusters on each shank (non noisy)
# 4. population statistics 
# 5. check post activation 

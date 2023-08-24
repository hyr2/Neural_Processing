#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 03:43:47 2023

@author: hyr2-office
"""

# Processess output from rICT_plot_fig1.m 

import scipy.io as sio # Import function to read data.
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import os
from natsort import natsorted
import pandas as pd
import sys
from Support import filterSignal_lowpass_new, ShowFFT, filter_Savitzky_slow
from scipy import signal

# Define the window size for the moving average filter
window_size = 25

input_dir = '/home/hyr2-office/Documents/Data/PureImaging_AcuteStrokes/RH7/'
plt.figure()
Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
rICT_infarcttissue = Data['rICT'][:,3]
rICT_fartissue = Data['rICT'][:,1]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > 700)[0][0]           # 500 seconds of data
rICT_infarcttissue[np.isnan(rICT_infarcttissue)] = 1
rICT_fartissue[np.isnan(rICT_fartissue)] = 1
rICT_infarcttissue = signal.savgol_filter(rICT_infarcttissue,130,1)
rICT_fartissue = signal.savgol_filter(rICT_fartissue,130,1)
plt.plot(time_ax[:indx_time],rICT_infarcttissue[:indx_time],linewidth = 2.8)
plt.plot(time_ax[:indx_time],rICT_fartissue[:indx_time],linewidth = 2.8)
plt.legend(['Core','Far'])
plt.ylim([0.61,1.38])
plt.axis('off')
plt.savefig(os.path.join(input_dir,'rICT_plot_filtered.svg'),format = 'svg')
plt.close()

plt.figure()
input_dir = '/home/hyr2-office/Documents/Data/PureImaging_AcuteStrokes/RH8/'
Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
rICT_fartissue = Data['rICT'][:,2]
rICT_infarcttissue = Data['rICT'][:,4]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > 700)[0][0]           # 500 seconds of data
rICT_fartissue[np.isnan(rICT_fartissue)] = 1
rICT_infarcttissue[np.isnan(rICT_infarcttissue)] = 1
rICT_infarcttissue = signal.savgol_filter(rICT_infarcttissue,130,1)
rICT_fartissue = signal.savgol_filter(rICT_fartissue,130,1)
plt.plot(time_ax[:indx_time],rICT_infarcttissue[:indx_time],linewidth = 2.8)
plt.plot(time_ax[:indx_time],rICT_fartissue[:indx_time],linewidth = 2.8)
plt.legend(['Core','Far'])
plt.ylim([0.61,1.38])
plt.axis('off')
plt.savefig(os.path.join(input_dir,'rICT_plot_filtered.svg'),format = 'svg')
plt.close()


plt.figure()
input_dir = '/home/hyr2-office/Documents/Data/PureImaging_AcuteStrokes/RH11/'
Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
rICT_fartissue = Data['rICT'][:,0]
rICT_infarcttissue = Data['rICT'][:,3]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > 700)[0][0]           # 500 seconds of data
rICT_fartissue[np.isnan(rICT_fartissue)] = 1
rICT_infarcttissue[np.isnan(rICT_infarcttissue)] = 1
rICT_infarcttissue = signal.savgol_filter(rICT_infarcttissue,130,1)
rICT_fartissue = signal.savgol_filter(rICT_fartissue,130,1)
plt.plot(time_ax[:indx_time],rICT_infarcttissue[:indx_time],linewidth = 2.8)
plt.plot(time_ax[:indx_time],rICT_fartissue[:indx_time],linewidth = 2.8)
plt.legend(['Core','Far'])
plt.ylim([0.61,1.38])
plt.axis('off')
plt.savefig(os.path.join(input_dir,'rICT_plot_filtered.svg'),format = 'svg')
plt.close()






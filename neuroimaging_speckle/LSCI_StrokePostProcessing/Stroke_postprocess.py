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
from Support import filterSignal_lowpass_new


input_dir = '/home/hyr2-office/Documents/Data/PureImaging_AcuteStrokes/RH8/'

Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
rICT_art = Data['rICT'][:,0]
rICT_infarcttissue = Data['rICT'][:,1]
rICT_fartissue = Data['rICT'][:,2]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > 450)[0][0]           # 500 seconds of data
# rICT_filtered = filter_Savitzky_slow(rICT_art)
rICT_infarcttissue[np.isnan(rICT_infarcttissue)] = 1
rICT_infarcttissue = filterSignal_lowpass_new(rICT_infarcttissue, Fs, axis_value = 0)
plt.plot(time_ax[:indx_time],rICT_infarcttissue[:indx_time])
rICT_fartissue[np.isnan(rICT_fartissue)] = 1
rICT_fartissue = filterSignal_lowpass_new(rICT_fartissue, Fs, axis_value = 0)
plt.plot(time_ax[:indx_time],rICT_fartissue[:indx_time])

plt.savefig(os.path.join(input_dir,'rICT_plot_filtered.svg'),format = 'svg')

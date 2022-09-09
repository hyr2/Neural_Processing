#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:30:58 2022

@author: Haad-Rathore
"""

import scipy.io as sio # Import function to read data.
from SupportC import *
import pylab
import os
from natsort import natsorted
import pandas as pd

# Source directory:
    # |
    # |__2021-12-06/
    # |__2021-12-07/
    # |__2021-12-09/
    #     .
    #     .
    #     .

# source_dir = input('Enter the source directory for one particular mouse:\n')
source_dir = '/home/hyr2/Documents/Data/BC7/'
source_dir_list = natsorted(os.listdir(source_dir))

# IOS_TS = {}  # empty dict for time series data (average of trials)
# IOS_ROI = {} # empty dict for summary info
# for name in source_dir_list:
#     folder_loc = os.path.join(source_dir, name)
#     if os.path.isdir(folder_loc):
#         folder_loc_mat = os.path.join(folder_loc,'IOS/Processed/mat_files/')
#         tmp = pd.DataFrame.from_dict(sio.loadmat(os.path.join(folder_loc_mat,'TimeS_dR.mat')))
#         IOS_TS
#         IOS_TS[name] = tmp

# IMPORTANT PARAMETERS:
lag = 14
threshold = 3
 

f1 = sio.loadmat('/home/hyr2/Documents/Data/BC7/2021-12-06/IOS/Processed/mat_files/TimeS_dR.mat')
f2 = sio.loadmat('/home/hyr2/Documents/Data/BC7/2021-12-07/IOS/Processed/mat_files/TimeS_dR.mat')
f3 = sio.loadmat('/home/hyr2/Documents/Data/BC7/2021-12-12/IOS/Processed/mat_files/TimeS_dR.mat')
f4 = sio.loadmat('/home/hyr2/Documents/Data/BC7/2021-12-17/IOS/Processed/mat_files/TimeS_dR.mat')
# f5 = pd.DataFrame.from_dict(sio.loadmat('/home/hyr2/Documents/Data/BC7/2021-12-24/IOS/Processed/mat_files/TimeS_dR.mat'))


# Loop over all dict items:
TS_480 = f1['TS_480']
TS_580 = f1['TS_580']
time_vec = np.linspace(0,13.5,75)
# time_vec = np.reshape(time_vec,[len(time_vec),])
TS_480_pk = {}
TS_580_pk = {}
for iter in range(0,6):
    TS_480_pk[iter] = thresholding_algo(TS_480[:,iter], lag=lag, threshold=threshold, influence = 0.01)
    TS_580_pk[iter] = thresholding_algo(TS_580[:,iter], lag=lag, threshold=threshold, influence = 0.01)
    
ROI_iter = 4
# Plotting
pylab.subplot(211)
pylab.plot(time_vec, TS_480[:,ROI_iter])

pylab.plot(time_vec,TS_480_pk[ROI_iter]["avgFilter"], color="cyan", lw=2)
pylab.plot(time_vec,TS_480_pk[ROI_iter]["avgFilter"] + threshold*TS_480_pk[ROI_iter]['stdFilter'], color="green", lw=2)
pylab.plot(time_vec,TS_480_pk[ROI_iter]["avgFilter"] - threshold*TS_480_pk[ROI_iter]['stdFilter'], color="green", lw=2)
pylab.subplot(212)
# pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.step(time_vec,TS_480_pk[ROI_iter]['signals'], color="red", lw=2)
# pylab.ylim(-1.5, 1.5)

# Plotting
pylab.figure()
pylab.subplot(211)
pylab.plot(time_vec, TS_580[:,ROI_iter])

pylab.plot(time_vec,TS_580_pk[ROI_iter]["avgFilter"], color="cyan", lw=2)
pylab.plot(time_vec,TS_580_pk[ROI_iter]["avgFilter"] + threshold*TS_580_pk[ROI_iter]['stdFilter'], color="green", lw=2)
pylab.plot(time_vec,TS_580_pk[ROI_iter]["avgFilter"] - threshold*TS_580_pk[ROI_iter]['stdFilter'], color="green", lw=2)
pylab.subplot(212)
# pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.step(time_vec,TS_580_pk[ROI_iter]['signals'], color="red", lw=2)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 02:42:19 2022

@author: hyr2-lap
"""
import scipy as sc
import numpy as np

def func_xcor_multi_trial(data_in_IOS,data_in_ePhys):
    [dim_trial,dim_time] = data_in_IOS.shape()
    xcorr_arr_out = np.zeros(dim_trial,dim_time)
    for iter_local in range(dim_trial):
        xcorr_local = sc.signal.correlate(data_in_IOS[iter_local,:], data_in_ePhys[iter_local,:])       #compute single trial cross-correlation
        xcorr_arr_out[iter_local,:] = xcorr_local
        
# Formatting the data

# Resampling 
[dim_trial,dim_time] = data_in_ePhys.shape()
data_in_IOS_resampled = np.zeros(dim_trial,dim_time)
for iter_local in range(dim_trial):
    arr_IOS_local = sc.signal.resample(data_in_IOS,dim_time)
    data_in_IOS_resampled[iter_local,:] = arr_IOS_local
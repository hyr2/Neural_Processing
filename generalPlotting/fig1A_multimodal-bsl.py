#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 19:53:23 2023

@author: hyr2
"""

from scipy import io
import numpy as np
from matplotlib import pyplot as plt
import os

# Input processed data from IOS matlab scripts (IOS_process_2.m)
source_dir = '/home/hyr2/Documents/Data/IOS_imaging/rh8/22-12-03'

wv_580_bsl1 = io.loadmat(os.path.join(source_dir,'580nm_processed.mat'))
wv_580_bsl1 = wv_580_bsl1['ROI_time_series_img']
wv_480_bsl1 = io.loadmat(os.path.join(source_dir,'480nm_processed.mat'))
wv_480_bsl1 = wv_480_bsl1['ROI_time_series_img']

ios_data = wv_580_bsl1[:,6]
# plt.plot(wv_580_bsl1[:,6])
# plt.legend(['A','B','C'])


# Input processed data from rickshaw_postprocessing.py (for firing rates of single units)




# Input processed data from LFP_core.py (for PSD analysis of LFP data)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:37:30 2022

@author: hyr2
"""

import os, sys
sys.path.append(os.path.join(os.getcwd(),'utils-mountainsort'))
sys.path.append(os.getcwd())
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy.signal as signal
import pandas as pd

source_dir1 = '/home/hyr2/Documents/Data/B-BC8/3-18-22/1/'
source_dir2 = '/home/hyr2/Documents/Data/B-BC8/3-18-22/2/'
source_dir3 = '/home/hyr2/Documents/Data/B-BC8/3-18-22/3/'

adjunct_dir = 'Processed/firing_rates/';

source_dir1 = os.path.join(source_dir1,adjunct_dir)
source_dir2 = os.path.join(source_dir2,adjunct_dir)
source_dir3 = os.path.join(source_dir3,adjunct_dir)

dir_list = natsorted(os.listdir(source_dir1))

# All whisker Array
arr_norm_FR_multi_whisker = np.zeros([3,4])
arr_abs_FR_multi_whisker = np.zeros([3,4])
arr_abs_FR_multi_whisker_best = np.zeros([3,4])

# Single whisker
arr_abs_FR_single_whisker = np.zeros([4,])
arr_norm_FR_single_whisker = np.zeros([4,])
arr_abs_FR_single_whisker_best = np.zeros([4,])
iter_local = 0
for filename in dir_list:
    
    path_file = os.path.join(source_dir1,filename)
    X = loadmat(path_file)
    bsl_fr = X['baseline_spike_rates']
    norm_fr = X['peak_normalized_firing_rate_during_stim']
    abs_fr = (norm_fr + 1)*bsl_fr - bsl_fr
    
    arr_abs_FR_single_whisker[iter_local] = np.mean(norm_fr)
    arr_norm_FR_single_whisker[iter_local] = np.mean(abs_fr)
    arr_abs_FR_single_whisker_best[iter_local] = np.amax(abs_fr)
    iter_local += 1
    
arr_norm_FR_multi_whisker[0,:] = arr_norm_FR_single_whisker
arr_abs_FR_multi_whisker[0,:] = arr_abs_FR_single_whisker
arr_abs_FR_multi_whisker_best[0,:] = arr_abs_FR_single_whisker_best
# Single whisker
arr_abs_FR_single_whisker = np.zeros([4,])
arr_norm_FR_single_whisker = np.zeros([4,])
iter_local = 0
for filename in dir_list:
    
    path_file = os.path.join(source_dir2,filename)
    X = loadmat(path_file)
    bsl_fr = X['baseline_spike_rates']
    norm_fr = X['peak_normalized_firing_rate_during_stim']
    abs_fr = (norm_fr + 1)*bsl_fr - bsl_fr
    
    arr_abs_FR_single_whisker[iter_local] = np.mean(norm_fr)
    arr_norm_FR_single_whisker[iter_local] = np.mean(abs_fr)
    arr_abs_FR_single_whisker_best[iter_local] = np.amax(abs_fr)
    iter_local += 1
    
arr_norm_FR_multi_whisker[1,:] = arr_norm_FR_single_whisker
arr_abs_FR_multi_whisker[1,:] = arr_abs_FR_single_whisker
arr_abs_FR_multi_whisker_best[1,:] = arr_abs_FR_single_whisker_best

# Single whisker
arr_abs_FR_single_whisker = np.zeros([4,])
arr_norm_FR_single_whisker = np.zeros([4,])
iter_local = 0
for filename in dir_list:
    
    path_file = os.path.join(source_dir3,filename)
    X = loadmat(path_file)
    bsl_fr = X['baseline_spike_rates']
    norm_fr = X['peak_normalized_firing_rate_during_stim']
    abs_fr = (norm_fr + 1)*bsl_fr - bsl_fr
    
    arr_abs_FR_single_whisker[iter_local] = np.mean(norm_fr)
    arr_norm_FR_single_whisker[iter_local] = np.mean(abs_fr)
    arr_abs_FR_single_whisker_best[iter_local] = np.amax(abs_fr)
    iter_local += 1
    
arr_norm_FR_multi_whisker[2,:] = arr_norm_FR_single_whisker
arr_abs_FR_multi_whisker[2,:] = arr_abs_FR_single_whisker
arr_abs_FR_multi_whisker_best[2,:] = arr_abs_FR_single_whisker_best
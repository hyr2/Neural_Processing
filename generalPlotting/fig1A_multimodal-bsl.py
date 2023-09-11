#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 19:53:23 2023

@author: hyr2
"""

from scipy import io
import numpy as np
from matplotlib import pyplot as plt
import os, pickle,sys
from scipy import signal
import seaborn as sns
sys.path.append('/home/hyr2/Documents/git/Neural_Processing/ePhys-LFP/')
from Support import *

# Input processed data from IOS matlab scripts (IOS_process_2.m)
source_dir = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/data/fig1A/'

wv_580_bsl1 = io.loadmat(os.path.join(source_dir,'580nm_processed.mat'))
wv_580_bsl1 = wv_580_bsl1['ROI_time_series_img']
wv_480_bsl1 = io.loadmat(os.path.join(source_dir,'480nm_processed.mat'))
wv_480_bsl1 = wv_480_bsl1['ROI_time_series_img']

ios_data = wv_580_bsl1[:,6]
time_axis_ios = np.linspace(0,13.5,75)


# Input processed data from rickshaw_postprocessing.py (for firing rates of single units)
pickle_file_path = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/data/fig1A/FR_all.pkl'
with open(pickle_file_path, 'rb') as file:
    # Load the data from the pickle file
    FR_dict = pickle.load(file)
time_axis_FR = FR_dict['time_axis']
avg_FR = FR_dict['FR']



# Input processed data from LFP_core.py (for PSD analysis of LFP data)
pickle_file_path = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/data/fig1A/shankA.pkl'
with open(pickle_file_path, 'rb') as file:
    # Load the data from the pickle file
    shankA = pickle.load(file)
pickle_file_path = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/data/fig1A/shankC.pkl'
with open(pickle_file_path, 'rb') as file:
    # Load the data from the pickle file
    shankC = pickle.load(file)
pickle_file_path = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/data/fig1A/shankD.pkl'
with open(pickle_file_path, 'rb') as file:
    # Load the data from the pickle file
    shankD = pickle.load(file)


time_axis_lfp = shankA['time_axis']
avg_trialsD = shankD['psd'][:,8]
avg_trialsD = signal.savgol_filter(avg_trialsD,220,3)
# avg_trialsA = shankA['psd'][:,5]

plt.plot(time_axis_ios,wv_580_bsl1[:,6]/np.amin(wv_580_bsl1[:,6]),linewidth = 2.5, color = '#949494')
plt.plot(time_axis_lfp,-avg_trialsD/np.amax(avg_trialsD), linewidth = 2.5, color = 'k')
# plt.plot(time_axis_lfp,0.00002 * signal.savgol_filter(avg_trialsA,220,3))
plt.plot(time_axis_FR,avg_FR[:,22]/np.amax(avg_FR[:,22]),linewidth = 2.75, color = 'k',linestyle = '--', dashes=(3, 4))
# plt.plot(time_axis_ios,wv_580_bsl1[:,6])
plt.xlim(0.5,13)
sns.despine()
filename_save = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/subfigures/all_modality.png'
plt.savefig(filename_save)


plt.plot(time_axis_ios,wv_580_bsl1[:,6]/np.amin(wv_580_bsl1[:,6]),linewidth = 2.5, color = '#949494')
plt.plot(time_axis_lfp,-avg_trialsD/np.amax(avg_trialsD), linewidth = 2.5, color = 'k')
# plt.plot(time_axis_lfp,0.00002 * signal.savgol_filter(avg_trialsA,220,3))
plt.plot(time_axis_FR,avg_FR[:,22]/np.amax(avg_FR[:,22]),linewidth = 2.75, color = 'k',linestyle = '--', dashes=(3, 4))
# plt.plot(time_axis_ios,wv_580_bsl1[:,6])
plt.xlim(2.3,3.5)
sns.despine()
# filename_save = '/home/hyr2/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/subfigures/all_modality.png'
# plt.savefig(filename_save)



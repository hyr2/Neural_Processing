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
sys.path.append('/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/ePhys-LFP/')
from Support import *
import time

# Input processed data from IOS matlab scripts (IOS_process_2.m)
source_dir = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/data/fig1A/'

t = time.localtime()
current_time = time.strftime("%m_%d_%Y_%H_%M", t) + '_fig1'
# output_folder = os.path.join(r'C:\Rice_2023\Data\Results',current_time)
output_folder = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
avg_trialsD = shankD['psd'][:,4]
avg_trialsD = signal.savgol_filter(avg_trialsD,69,1)
# list_lfp_avg = []
# legend_axis = np.arange(0,avg_trialsD.shape[1])
# for iter_i in range(10):
#     list_lfp_avg.append(signal.savgol_filter(avg_trialsD[:,iter_i],69,1))
#     # list_lfp_avg.append(avg_trialsD[:,iter_i])
#     plt.plot(time_axis_lfp,np.array(list_lfp_avg[iter_i],dtype = float))
# plt.legend(legend_axis)

# avg_trialsA = shankA['psd'][:,5]
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)
# twin1 = ax.twinx()
# p1, = ax.plot(time_axis_ios, wv_580_bsl1[:,6]*100, color = '#949494' , label="IOS",linewidth = 2.5)
# p2, = ax.plot(time_axis_lfp, avg_trialsD, linewidth = 2.5, color = 'k', label = 'LFP')
# p3, = twin1.plot(time_axis_FR,avg_FR,linewidth = 1, color = 'b')
# twin1.set_xlim(left = 0.5,right = 13)
# twin1.set_ylim(bottom = -55.4,top = 45)


ax.plot(time_axis_ios,wv_580_bsl1[:,6]*100*20,linewidth = 2.5, color = '#1f77b4',alpha = 0.75)  # IOS
# ax.plot(time_axis_lfp, 25*avg_trialsD, linewidth = 2.5, color = 'b')   # lfp  (Lan suggested to remove LFP)
ax.plot(time_axis_FR,avg_FR,linewidth = 2, color = '#d00808',alpha = 0.75)    # FR
# plt.plot(time_axis_ios,wv_580_bsl1[:,6])
ax.set_xlim(0.5,13)
ax.set_ylim(-30,43.4)
# sns.despine()
ax.axis('off')
filename_save = os.path.join(output_folder,'all_modality.png')
# '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/subfigures/all_modality.png'
plt.savefig(filename_save)


# plt.plot(time_axis_ios,wv_580_bsl1[:,6]/np.amin(wv_580_bsl1[:,6]),linewidth = 2.5, color = '#949494')
# plt.plot(time_axis_lfp,-avg_trialsD/np.amax(avg_trialsD), linewidth = 2.5, color = 'k')
# # plt.plot(time_axis_lfp,0.00002 * signal.savgol_filter(avg_trialsA,220,3))
# plt.plot(time_axis_FR,avg_FR[:,22]/np.amax(avg_FR[:,22]),linewidth = 2.75, color = 'k',linestyle = '--', dashes=(3, 4))
# # plt.plot(time_axis_ios,wv_580_bsl1[:,6])
# plt.xlim(2.3,3.5)
# sns.despine()
# filename_save = '/home/hyr2/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig1/subfigures/all_modality.png'
# plt.savefig(filename_save)



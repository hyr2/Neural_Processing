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
t_thresh = 650

input_dir = '/home/hyr2-office/Documents/Data/PureImaging_AcuteStrokes/RH7/'
fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
fig.tight_layout(pad = 5)
Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
rICT_infarcttissue = Data['rICT'][:,0]
rICT_periinfarct = Data['rICT'][:,2]
rICT_fartissue = Data['rICT'][:,3]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > t_thresh)[0][0]           # 12 min of data
rICT_infarcttissue[np.isnan(rICT_infarcttissue)] = 1
rICT_fartissue[np.isnan(rICT_fartissue)] = 1
rICT_periinfarct[np.isnan(rICT_periinfarct)] = 1
rICT_infarcttissue = signal.savgol_filter(rICT_infarcttissue,130,1)
rICT_fartissue = signal.savgol_filter(rICT_fartissue,130,1)
rICT_periinfarct = signal.savgol_filter(rICT_periinfarct,130,1)
axes.plot(time_ax[:indx_time],rICT_infarcttissue[:indx_time],linewidth = 3.5,c = 'g')
axes.plot(time_ax[:indx_time],rICT_fartissue[:indx_time],linewidth = 3.5,c = 'b')
axes.plot(time_ax[:indx_time],rICT_periinfarct[:indx_time],linewidth = 3.5,c = '#4fc4cd')
# axes.legend(['Core','Far','Peri-Infarct'],loc = 1)
# axes.set_ylim([0.61,1.38])
axes.set_xlabel('')
axes.set_ylabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True)         # ticks along the top edge are off
axes.legend().set_visible(False)
sns.despine(ax=axes, left=False, bottom=False, trim=False)
axes.set_ylim(0.5,2.25)
axes.set_xlim(0,650)
axes.set_yticks([0.5,0.75,1,1.25,1.5,1.75,2])
fig.set_size_inches(5,2.5,forward=True)
fig.savefig(os.path.join(input_dir,'rICT_plot_filtered_rh7.png'),format = 'png')
plt.close(fig)


fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
fig.tight_layout(pad = 5)
input_dir = '/home/hyr2-office/Documents/Data/PureImaging_AcuteStrokes/RH8/'
Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
rICT_infarcttissue = Data['rICT'][:,0]
rICT_2 = Data['rICT'][:,2]
rICT_3 = Data['rICT'][:,3]
# rICT_4 = Data['rICT'][:,3]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > t_thresh)[0][0]           # 12 min of data
rICT_2[np.isnan(rICT_2)] = 1
rICT_infarcttissue[np.isnan(rICT_infarcttissue)] = 1
rICT_3[np.isnan(rICT_3)] = 1
rICT_infarcttissue = signal.savgol_filter(rICT_infarcttissue,130,1)
rICT_2 = signal.savgol_filter(rICT_2,130,1)
rICT_3 = signal.savgol_filter(rICT_3,130,1)
axes.plot(time_ax[:indx_time],rICT_infarcttissue[:indx_time],linewidth = 3.5, c = 'g')
axes.plot(time_ax[:indx_time],rICT_2[:indx_time],linewidth = 3.5, c = 'b')
axes.plot(time_ax[:indx_time],rICT_3[:indx_time],linewidth = 3.5, c = '#4fc4cd')
axes.set_xlabel('')
axes.set_ylabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True)         # ticks along the top edge are off
axes.legend().set_visible(False)
sns.despine(ax=axes, left=False, bottom=False, trim=False)
axes.set_ylim(0.5,1.4)
axes.set_xlim(0,650)
axes.set_yticks([0.5,0.75,1,1.25])
fig.set_size_inches(5,2.5,forward=True)
fig.savefig(os.path.join(input_dir,'rICT_plot_filtered_rh8.png'),format = 'png')
plt.close(fig)

# Selected for Paper
fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
fig.tight_layout(pad = 5)
input_dir = '/home/hyr2-office/Documents/Data/PureImaging_AcuteStrokes/RH11/'
Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
rICT_infarcttissue = Data['rICT'][:,0]
rICT_periinfarct = Data['rICT'][:,2]
rICT_fartissue = Data['rICT'][:,4]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > t_thresh)[0][0]           # 12 min of data
rICT_fartissue[np.isnan(rICT_fartissue)] = 1
rICT_infarcttissue[np.isnan(rICT_infarcttissue)] = 1
rICT_periinfarct[np.isnan(rICT_periinfarct)] = 1
rICT_infarcttissue = signal.savgol_filter(rICT_infarcttissue,130,1)
rICT_fartissue = signal.savgol_filter(rICT_fartissue,130,1)
rICT_periinfarct = signal.savgol_filter(rICT_periinfarct,130,1)
axes.plot(time_ax[:indx_time],rICT_infarcttissue[:indx_time],linewidth = 3.5,c = 'g')
axes.plot(time_ax[:indx_time],rICT_fartissue[:indx_time],linewidth = 3.5, c = 'b')
axes.plot(time_ax[:indx_time],rICT_periinfarct[:indx_time],linewidth = 3.5, c = '#4fc4cd')
axes.legend().set_visible(False)
axes.set_xlabel('')
axes.set_ylabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True)         # ticks along the top edge are off
sns.despine(ax=axes, left=False, bottom=False, trim=False)
axes.set_yticks([0.5,0.75,1,1.25])
axes.set_ylim(0.5,1.4)
fig.set_size_inches(5,2.5,forward=True)
fig.savefig(os.path.join(input_dir,'rICT_plot_filtered_rh11.png'),format = 'png')
plt.close(fig)



fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
fig.tight_layout(pad = 5)
input_dir = '/home/hyr2-office/Documents/Data/PureImaging_AcuteStrokes/BH2/'
Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
rICT_infarcttissue = Data['rICT'][:,0]
rICT_periinfarct = Data['rICT'][:,1]
# rICT_fartissue = Data['rICT'][:,3]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > t_thresh)[0][0]           # 12 min of data
# rICT_fartissue[np.isnan(rICT_fartissue)] = 1
rICT_infarcttissue[np.isnan(rICT_infarcttissue)] = 1
rICT_periinfarct[np.isnan(rICT_periinfarct)] = 1
rICT_infarcttissue = signal.savgol_filter(rICT_infarcttissue,130,1)
# rICT_fartissue = signal.savgol_filter(rICT_fartissue,130,1)
rICT_periinfarct = signal.savgol_filter(rICT_periinfarct,130,1)
axes.plot(time_ax[:indx_time],rICT_infarcttissue[:indx_time],linewidth = 3.5)
# axes.plot(time_ax[:indx_time],rICT_fartissue[:indx_time],linewidth = 3.5)
axes.plot(time_ax[:indx_time],rICT_periinfarct[:indx_time],linewidth = 3.5)
# axes.legend(['Core','Far','Peri-Infarct'],loc = 1)
# axes.set_ylim([0.61,1.38])
axes.set_xlabel('')
axes.set_ylabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True)         # ticks along the top edge are off

sns.despine(ax=axes, left=False, bottom=False, trim=False)
axes.set_ylim(0.4,1.5)
axes.set_xlim(0,650)
# axes.set_yticks([0.5,1])
fig.set_size_inches(4.5,2.2,forward=True)
fig.savefig(os.path.join(input_dir,'rICT_plot_filtered_bh2.svg'),format = 'svg')
plt.close(fig)

import matplotlib as mpl
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 13}

mpl.rc('font', **font)

fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
fig.tight_layout(pad = 5)
input_dir = '/home/hyr2/Documents/git/Thesis/9-24-2021-stroke(1)/'
Data = sio.loadmat(os.path.join(input_dir,'data.mat'))
time_ax = np.squeeze(Data['t'])
# rICT_infarcttissue = Data['rICT'][:,0]
rICT_1 = Data['rICT'][:,0]
rICT_2 = Data['rICT'][:,1]
rICT_3 = Data['rICT'][:,2]
rICT_4 = Data['rICT'][:,3]
Fs = 1/np.mean(np.diff(time_ax))
indx_time = np.where(time_ax > t_thresh)[0][0]           # 12 min of data
rICT_1[np.isnan(rICT_1)] = 1
rICT_2[np.isnan(rICT_2)] = 1
rICT_3[np.isnan(rICT_3)] = 1
rICT_4[np.isnan(rICT_4)] = 1
rICT_1 = signal.savgol_filter(rICT_1,130,1)
rICT_2 = signal.savgol_filter(rICT_2,130,1)
rICT_3 = signal.savgol_filter(rICT_3,130,1)
rICT_4 = signal.savgol_filter(rICT_4,130,1)
axes.plot(time_ax[:indx_time],rICT_1[:indx_time],linewidth = 3.5, c = '#A22222')
axes.plot(time_ax[:indx_time],rICT_2[:indx_time],linewidth = 3.5, c = '#28A828')
axes.plot(time_ax[:indx_time],rICT_3[:indx_time],linewidth = 3.5, c = '#2f2faf')
axes.plot(time_ax[:indx_time],rICT_4[:indx_time],linewidth = 3.5, c = '#23a3a3')
axes.set_xlabel('')
axes.set_ylabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True)         # ticks along the top edge are off
axes.legend().set_visible(False)
# axes.legend(['1','2','3','4'])
sns.despine(ax=axes, left=False, bottom=False, trim=False)
axes.set_ylim(0.25,1.6)
# axes.set_xlim(0,1000)
axes.set_yticks([0.25,0.5,0.75,1,1.25,1.5])
fig.set_size_inches(5,3.5,forward=True)
fig.savefig(os.path.join(input_dir,'rICT_plot_filtered_.png'),format = 'png')
plt.close(fig)






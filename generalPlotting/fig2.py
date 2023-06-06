#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 00:01:55 2023

@author: hyr2-office
"""

# raster plots
# activated and suppressed neuron firing rate time series

import sys
sys.path.append('/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/ePhys-Spikes/Spike-Sorting-Code/post_msort_processing/')
import numpy as np
from matplotlib import pyplot as plt
import os
from utils.read_mda import readmda
from scipy.io import loadmat
from Support import filter_Savitzky_fast


filename_save = '/home/hyr2-office/Documents/Paper/Single-Figures-SVG/Fig2/subfigures/'


def raster_all_trials(firing_stamp, t_trial_start, trial_duration_in_samples, window_in_samples):
    # returns the spike bin counts; need to manuaaly divide result by window length to get real firing rate
    n_windows_in_trial = int(np.ceil(trial_duration_in_samples/window_in_samples))
    bin_edges = np.arange(0, window_in_samples*n_windows_in_trial+1, step=window_in_samples)
    n_trials = t_trial_start.shape[0]
    # print(n_trials,Ntrials)
    # firing_rate_series_by_trial = np.zeros((n_trials, n_windows_in_trial))
    raster_series = []
    for i in range(n_trials):
        trial_firing_mask = np.logical_and(firing_stamp>=t_trial_start[i], firing_stamp<=(t_trial_start[i]+trial_duration_in_samples))
        trial_firing_stamp = firing_stamp[trial_firing_mask] - t_trial_start[i]
        trial_firing_stamp = trial_firing_stamp/trial_duration_in_samples * 13.5 # time axis raster plot
        if trial_firing_stamp.shape[0]==0:
            continue
        raster_series.append(trial_firing_stamp)
    return raster_series

def FR_all_trials(firing_stamp, t_trial_start, trial_duration_in_samples, window_in_samples):
    # returns the spike bin counts; need to manuaaly divide result by window length to get real firing rate
    n_windows_in_trial = int(np.ceil(trial_duration_in_samples/window_in_samples))
    bin_edges = np.arange(0, window_in_samples*n_windows_in_trial+1, step=window_in_samples)
    n_trials = t_trial_start.shape[0]
    # print(n_trials,Ntrials)
    firing_rate_series_by_trial = np.zeros((n_trials, n_windows_in_trial))
    for i in range(n_trials):
        trial_firing_mask = np.logical_and(firing_stamp>=t_trial_start[i], firing_stamp<=(t_trial_start[i]+trial_duration_in_samples))
        trial_firing_stamp = firing_stamp[trial_firing_mask] - t_trial_start[i]
        # trial_firing_stamp = trial_firing_stamp/trial_duration_in_samples * 13.5 # time axis raster plot
        if trial_firing_stamp.shape[0]==0:
            continue
        tmp_hist, _ = np.histogram(trial_firing_stamp, bin_edges)   # The firing rate series for each trial for a single cluster
        firing_rate_series_by_trial[i,:] = tmp_hist   
    return firing_rate_series_by_trial


# Raster plotting
Firings_bsl = readmda('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/firings.mda')
trials_bsl = loadmat('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/trials_times.mat')['t_trial_start'].squeeze()
firings_bsl_351 = Firings_bsl[1,Firings_bsl[2,:] == 351]
firings_bsl_10 = Firings_bsl[1,Firings_bsl[2,:] == 10]
firings_bsl_nr = Firings_bsl[1,Firings_bsl[2,:] == 184]

trial_duration_in_samples = 13.5 * 30e3
window_in_samples = 50e-3 * 30e3
firing_rate_series_351 = raster_all_trials(
    firings_bsl_351, 
    trials_bsl, 
    trial_duration_in_samples, 
    window_in_samples
    )
firing_rate_series_10 = raster_all_trials(
    firings_bsl_10, 
    trials_bsl, 
    trial_duration_in_samples, 
    window_in_samples
    )
firing_rate_series_nr = raster_all_trials(
    firings_bsl_nr, 
    trials_bsl, 
    trial_duration_in_samples, 
    window_in_samples
    )
# plt.plot(firing_raster,y1,color = 'black',marker = ".",linestyle = 'None')
y_local = 0
for iter_t in range(len(firing_rate_series_10)):
    raster_local = firing_rate_series_10[iter_t]
    y_local = iter_t + np.ones(raster_local.shape)
    plt.plot(raster_local,y_local,color = 'blue',marker = "o",linestyle = 'None', markersize = 3)
plt.xlim(2,3.5)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-cluster10_raster.svg'),format = 'svg')
y_local = 0
plt.figure()
for iter_t in range(len(firing_rate_series_351)):
    raster_local = firing_rate_series_351[iter_t]
    y_local = iter_t + np.ones(raster_local.shape)
    plt.plot(raster_local,y_local,color = 'red',marker = "o",linestyle = 'None',markersize = 3)
plt.xlim(2,3.5)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-cluster315_raster.svg'),format = 'svg')
plt.figure()
for iter_t in range(len(firing_rate_series_nr)):
    raster_local = firing_rate_series_nr[iter_t]
    y_local = iter_t + np.ones(raster_local.shape)
    plt.plot(raster_local,y_local,color = 'blue',marker = "o",linestyle = 'None',markersize = 3)
plt.xlim(2,3.5)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-cluster184_raster.svg'),format = 'svg')


# Firing rate average over trial (single cluster)
firing_rate_series_351 = FR_all_trials(
    firings_bsl_351, 
    trials_bsl, 
    trial_duration_in_samples, 
    window_in_samples
    )
firing_rate_series_10 = FR_all_trials(
    firings_bsl_10, 
    trials_bsl, 
    trial_duration_in_samples, 
    window_in_samples
    )
firing_rate_series_nr = FR_all_trials(
    firings_bsl_nr, 
    trials_bsl, 
    trial_duration_in_samples, 
    window_in_samples
    )
x_axis = np.linspace(0,13.5,firing_rate_series_10.shape[1])
plt.figure()
plt.plot(x_axis,filter_Savitzky_fast(np.mean(firing_rate_series_10,axis = 0)),'b',linewidth = 3)
plt.axis('off')
plt.xlim(2,3.5)
plt.ylim(0,1.5)
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-cluster10_FR.svg'),format = 'svg')
plt.figure()
plt.plot(x_axis,filter_Savitzky_fast(np.mean(firing_rate_series_351,axis = 0)),'r',linewidth = 3)
plt.axis('off')
plt.xlim(2,3.5)
plt.ylim(0,1.5)
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-cluster351_FR.svg'),format = 'svg')
plt.figure()
plt.plot(x_axis,filter_Savitzky_fast(np.mean(firing_rate_series_nr,axis = 0)),'b',linewidth = 3)
plt.axis('off')
plt.xlim(2,3.5)
plt.ylim(0,1.5)
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-cluster184_FR.svg'),format = 'svg')
# filename_save = '/home/hyr2-office/Documents/Paper/Single-Figures-SVG/Fig2/subfigures/'

# Here plotting the waveforms of these clusters
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus10']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.figure()
plt.plot(x_axis*1000,y11,'b',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-23_10.svg'),format = 'svg')

# Here plotting the waveforms of these clusters
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus184']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.figure()
plt.plot(x_axis*1000,y11,'b',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-23_184.svg'),format = 'svg')

# Here plotting the waveforms of these clusters
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus351']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.figure()
plt.plot(x_axis*1000,y11,'r',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-23_351.svg'),format = 'svg')

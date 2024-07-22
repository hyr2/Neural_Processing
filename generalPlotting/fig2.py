#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 00:01:55 2023

@author: hyr2-office
"""

# raster plots
# activated and suppressed neuron firing rate time series

import sys, json
sys.path.append('/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/ePhys-Spikes/Spike-Sorting-Code/post_msort_processing/')
import numpy as np
from matplotlib import pyplot as plt
import os
from utils.read_mda import readmda
from scipy.io import loadmat
from Support import filter_Savitzky_fast
import pandas as pd
import time




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

source_dir = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_bc7/21-12-09/'
t = time.localtime()
current_time = time.strftime("%m_%d_%Y_%H_%M", t) + '_fig2D'
# output_folder = os.path.join(r'C:\Rice_2023\Data\Results',current_time)
filename_save = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)
if not os.path.exists(filename_save):
    os.makedirs(filename_save)

# Raster plotting
Firings_bsl = readmda(os.path.join(source_dir,'firings_clean_merged.mda'))
file_pre_ms = os.path.join(source_dir,'pre_MS.json')
trials_bsl = loadmat(os.path.join(source_dir,'trials_times.mat'))['t_trial_start'].squeeze()
geom = pd.read_csv(os.path.join(source_dir,'geom.csv'),header=None).to_numpy()
templates = readmda(os.path.join(source_dir,'templates_clean_merged.mda'))
with open(file_pre_ms, 'r') as f:
  data_pre_ms = json.load(f)
F_SAMPLE = float(data_pre_ms['SampleRate'])
CHMAP2X16 = bool(data_pre_ms['ELECTRODE_2X16'])      # this affects how the plots are generated
Num_chan = int(data_pre_ms['NumChannels'])
Notch_freq = float(data_pre_ms['Notch filter'])
Fs = float(data_pre_ms['SampleRate'])
stim_start_time = float(data_pre_ms['StimulationStartTime'])
n_stim_start = int(Fs * stim_start_time)
Ntrials = int(data_pre_ms['NumTrials'])
stim_end_time = stim_start_time + float(data_pre_ms['StimulationTime'])
time_seq = float(data_pre_ms['SequenceTime'])
Seq_perTrial = float(data_pre_ms['SeqPerTrial'])
total_time = time_seq * Seq_perTrial
print('Each sequence is: ', time_seq, 'sec')
time_seq = int(np.ceil(time_seq * Fs/2))

# number of clusters
num_clusters = np.unique(Firings_bsl[2,:]).shape[0]
trial_duration_in_samples = total_time * F_SAMPLE
time_window = 20e-3
window_in_samples = time_window * F_SAMPLE

# plot and save waveform profile for each cluster in the curated data
for iter_l in range(num_clusters):
    filename_save_local = os.path.join(filename_save,'FR' + str(iter_l+1) + '.png')
    firing_local = Firings_bsl[1,Firings_bsl[2,:] == iter_l+1]
    prim_ch_local = np.squeeze(np.unique(Firings_bsl[0,Firings_bsl[2,:] == iter_l+1])) - 1           # subtract 1 since conventionally prim_channels in .mda start from 1 instead of 0
    if prim_ch_local.size != 1:
        raise ValueError("Issue in primary channel. Cluster has more than one primary channel!\n")
    
    y = templates[int(prim_ch_local),:,iter_l]
    x_axis = np.linspace(-1,1,y.shape[0])    # 2 ms 
    plt.plot(y, color = 'k', linewidth = 3.5)
    loc_local = geom[int(prim_ch_local),:]
    plt.title('ClusterID:' + str(iter_l+1) + str(loc_local))
    plt.text(80, -75, firing_local.shape[0] , fontsize=12, color='red', ha='center', va='center')
    plt.savefig(filename_save_local,dpi = 100, format = 'png')
    plt.clf()
    plt.close()



for iter_clust in range(num_clusters):
    
    firings_bsl_nr = Firings_bsl[1,Firings_bsl[2,:] == iter_clust+1]  #12 with firings.mda
    
    firing_rate_series_nr = raster_all_trials(
        firings_bsl_nr, 
        trials_bsl, 
        trial_duration_in_samples, 
        window_in_samples
        )
    # plt.plot(firing_raster,y1,color = 'black',marker = ".",linestyle = 'None')
    fig,ax = plt.subplots(1,1)
    y_local = 0
    for iter_t in range(len(firing_rate_series_nr)):
        raster_local = firing_rate_series_nr[iter_t]
        y_local = iter_t + np.ones(raster_local.shape)
        ax.plot(raster_local,y_local,color = 'k',marker = "o",linestyle = 'None', markersize = 3,alpha = 0.5,markeredgewidth = 0)
    ax.set_xlim(2,3.5)
    fig.set_size_inches(7,4)
    ax.axis('off')
    fig.savefig(os.path.join(filename_save,f'cluster_{iter_clust+1}_raster.png'),format = 'png',dpi = 350)
    plt.close(fig)
    
    
    # plt.plot(firing_raster,y1,color = 'black',marker = ".",linestyle = 'None')
    firing_rate_series_nr = FR_all_trials(firings_bsl_nr, 
    trials_bsl, 
    trial_duration_in_samples, 
    window_in_samples
    )
    fig,ax = plt.subplots(1,1)
    x_axis = np.linspace(0,13.5,firing_rate_series_nr.shape[1])
    ax.plot(x_axis,filter_Savitzky_fast(np.mean(firing_rate_series_nr/time_window,axis = 0)),'k',linewidth = 4)
    ax.set_xlim(2,3.5)
    fig.set_size_inches(3.5,2)
    # ax.axis('off')
    fig.savefig(os.path.join(filename_save,f'cluster_{iter_clust+1}_FR.png'),format = 'png',dpi = 350)
    plt.close(fig)
    
    
    
    
firings_bsl_351 = Firings_bsl[1,Firings_bsl[2,:] == 37]
firings_bsl_10 = Firings_bsl[1,Firings_bsl[2,:] == 38]
firings_bsl_nr = Firings_bsl[1,Firings_bsl[2,:] == 39]  #12 with firings.mda


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
fig,ax = plt.subplots(1,1)
y_local = 0
for iter_t in range(len(firing_rate_series_10)):
    raster_local = firing_rate_series_10[iter_t]
    y_local = iter_t + np.ones(raster_local.shape)
    ax.plot(raster_local,y_local,color = 'k',marker = "o",linestyle = 'None', markersize = 8,alpha = 0.5,markeredgewidth = 0)
ax.set_xlim(2,3.5)
ax.axis('off')
fig.savefig(os.path.join(filename_save,'rh7-10-17-22-clusterA_raster.svg'),format = 'svg')
y_local = 0
fig,ax = plt.subplots(1,1)
for iter_t in range(len(firing_rate_series_351)):
    raster_local = firing_rate_series_351[iter_t]
    y_local = iter_t + np.ones(raster_local.shape)
    ax.plot(raster_local,y_local,color = 'red',marker = "o",linestyle = 'None',markersize = 6,alpha = 0.5, markeredgewidth = 0)
ax.set_xlim(2,3.5)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-clusterB_raster.svg'),format = 'svg')
fig,ax = plt.subplots(1,1)
for iter_t in range(len(firing_rate_series_nr)):
    raster_local = firing_rate_series_nr[iter_t]
    y_local = iter_t + np.ones(raster_local.shape)
    ax.plot(raster_local,y_local,color = 'blue',marker = "o",linestyle = 'None',markersize = 3)
ax.set_xlim(2,3.5)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-clusterC_raster.svg'),format = 'svg')


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
plt.ylim(0,2)
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-clusterA_FR.svg'),format = 'svg')
plt.figure()
plt.plot(x_axis,filter_Savitzky_fast(np.mean(firing_rate_series_351,axis = 0)),'r',linewidth = 3)
plt.axis('off')
plt.xlim(2,3.5)
plt.ylim(0,2)
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-clusterB_FR.svg'),format = 'svg')
plt.figure()
plt.plot(x_axis,filter_Savitzky_fast(np.mean(firing_rate_series_nr,axis = 0)),'b',linewidth = 3)
plt.axis('off')
plt.xlim(2,3.5)
plt.ylim(0,2)
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-clusterC_FR.svg'),format = 'svg')
# filename_save = '/home/hyr2-office/Documents/Paper/Single-Figures-SVG/Fig2/subfigures/'



y1 = []     # baseline average waveforms
ywk5 = []   # for wk 3

# Here plotting the waveforms of these clusters
# ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/all_waveforms_by_cluster.npz',allow_pickle = True)
cluster_ID = [1,3,5,125,44]  # For baseline 12-09-21 BC7
colors_clusters_bsl = ['#00008B','#FF2400','#00008B','#FF2400','#FF2400']
for iter_l in cluster_ID:
    pri_ch = np.unique(Firings_bsl[0,Firings_bsl[2,:] == iter_l]) # primary channel (starts from 1)
    pri_ch = np.squeeze(pri_ch) - 1
    y1.append(templates[int(pri_ch),:,iter_l-1])
    # y1 = ss['clus10']
    # y11 = np.mean(y1,axis = 0)
    # x_axis = np.linspace(0,3.33e-3,num = 100)
    # plt.figure()
    # plt.plot(x_axis*1000,y1,'r',linewidth = 2.9)
    # plt.axis('off')
    # plt.savefig(os.path.join(filename_save,'bc7-12-09-21_'+str(cluster_ID)+'.svg'),format = 'svg')
    
# For spike overlap plots
# filename_save = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_combined/'
source_dir = '/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/tmp_bc7/21-12-31/'
Firings_bsl = readmda(os.path.join(source_dir,'firings_clean_merged.mda'))
file_pre_ms = os.path.join(source_dir,'pre_MS.json')
trials_bsl = loadmat(os.path.join(source_dir,'trials_times.mat'))['t_trial_start'].squeeze()
geom = pd.read_csv(os.path.join(source_dir,'geom.csv'),header=None).to_numpy()
templates = readmda(os.path.join(source_dir,'templates_clean_merged.mda'))
cluster_ID = [1,3,5,66,31]
colors_clusters = ['#6F8FAF','#F88379','#6F8FAF','#F88379','#F88379']
for iter_l in cluster_ID:
    pri_ch = np.unique(Firings_bsl[0,Firings_bsl[2,:] == iter_l]) # primary channel (starts from 1)
    pri_ch = np.squeeze(pri_ch) - 1
    ywk5.append(templates[int(pri_ch),:,iter_l-1])
    # y1 = ss['clus10']
    # y11 = np.mean(y1,axis = 0)
    # x_axis = np.linspace(0,3.33e-3,num = 100)
    # plt.figure()
    # plt.plot(x_axis*1000,y1,'r',linewidth = 2.9)
    # plt.axis('off')
    # plt.savefig(os.path.join(filename_save,'bc7-12-09-21_'+str(cluster_ID)+'.svg'),format = 'svg')

# Here plotting the waveforms of these clusters     (manually select red or blue for each depending on pyr or narrow interneuron)
# color = '#f30101' and #f69c9c for pyr
# color = '#0302f6' and #aaa9f5 for FS/PV
x_axis = np.linspace(0,3.33e-3,num = 100) 
for iter_l in range(len(cluster_ID)):
    f, a = plt.subplots(1,2)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = a.flatten()
    axes[1].plot(x_axis*1000,ywk5[iter_l],color = colors_clusters[iter_l],linewidth = 3.2)      # Week5 12-31 Post Stroke
    axes[0].plot(x_axis*1000,y1[iter_l],color = colors_clusters_bsl[iter_l],linewidth = 3.2)    # Baseline 12-09
    
    # axes[0].set_xlim([])
    axes[0].set_ylim([-193,54])
    # axes[1].set_xlim([])
    axes[1].set_ylim([-193,54])
    axes[1].axis('off')
    axes[0].axis('off')
    
    f.set_size_inches((10, 6), forward=False)
    plt.savefig(os.path.join(filename_save,'bc7-bsl_wk5_compared'+str(iter_l)+'.svg'),format = 'svg')
    plt.close(f)


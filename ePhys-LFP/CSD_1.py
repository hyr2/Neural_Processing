# -*- coding: utf-8 -*-
"""
Created on Sun July 19 00:20:16 2021
@author: Haad-Rathore
"""

from scipy.io import loadmat       # Import function to read data.
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from scipy import signal, integrate, stats
import sys, os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from natsort import natsorted
from Support import *

source_dir = input('Enter the source directory: \n')
output_dir = os.path.join(source_dir,'Processed','CSD')
dir_chan_list = os.path.join(source_dir,'chan_list.xlsx')
dir_chan_map = os.path.join(source_dir,'chan_map_1x32_128ch.xlsx')
dir_Bin2Trials = os.path.join(source_dir,'Bin2Trials')
dir_expsummary = os.path.join(source_dir,'exp_summary.xlsx')

os.makedirs(output_dir)


# Extracting data from summary file .xlsx
df_exp_summary = pd.read_excel(dir_expsummary)
arr_exp_summary = df_exp_summary.to_numpy()
Num_chan = arr_exp_summary[0,0]         # Number of channels
Notch_freq = arr_exp_summary[0,1]       # Notch frequencey selected (in Hz)
Fs = arr_exp_summary[0,2]               # Sampling freq (in Hz)
stim_start_time = arr_exp_summary[4,1]  # Stimulation start
n_stim_start = int(Fs * stim_start_time)# Stimulation start time in samples
Ntrials = arr_exp_summary[2,4]          # Number of trials
stim_end_time = arr_exp_summary[2,1] + stim_start_time  # End time of stimulation
time_seq = arr_exp_summary[4,0]         # Time of one sequence in seconds
Seq_perTrial =  arr_exp_summary[2,3]    # Number of sequences per trial
total_time = time_seq * Seq_perTrial    # Total time of the trial
print('Each sequence is: ', time_seq, 'sec')
time_seq = int(np.ceil(time_seq * Fs/2) * 2)                # Time of one sequence in samples (rounded up to even)

# --------------------- SET THESE PARAMETERS ------------------------------
time_window = 150e-3                    # Selecting a 400 ms time window
n_time_window = int(time_window * Fs)   # Time in samples
n_chan = 128;                           # Total channels in the device
n_density = 32                          # Number of channels on a single shank
spacing = 25e-6                         # Spacing in microns b/w electrodes

# Extracting channel mapping info
df_chanMap = pd.read_excel(dir_chan_map,usecols=[0,1,2,3],header = None,sheet_name = 2)
arr_chanMap = df_chanMap.to_numpy()                 # 4 shank device 1x32 channels on each shank
df_chanList = pd.read_excel(dir_chan_list,header = 0)
chan_list = df_chanList.to_numpy()
chan_list = np.reshape(chan_list,(Num_chan,))

# First channel only 
filename = os.path.join(dir_Bin2Trials,'Chan0.csv')
df_chan = pd.read_csv(filename,dtype = np.single)
arr_chan = df_chan.to_numpy()
t = arr_chan[:,-1]
adc_data = arr_chan[:,-2]

# Creating 3D matrices to store the data
Adata_2D = np.zeros((arr_chanMap.shape[0],3*n_time_window),dtype = np.single)
Adata_2D[:] = np.nan
Bdata_2D = np.zeros((arr_chanMap.shape[0],3*n_time_window),dtype = np.single)
Bdata_2D[:] = np.nan
Cdata_2D = np.zeros((arr_chanMap.shape[0],3*n_time_window),dtype = np.single)
Cdata_2D[:] = np.nan
Ddata_2D = np.zeros((arr_chanMap.shape[0],3*n_time_window),dtype = np.single)
Ddata_2D[:] = np.nan


# Shanks (rejecting shanks with 6 channels missing)
n_shankA = 0
n_shankB = 0
n_shankC = 0
n_shankD = 0
for iter in chan_list:
    loc_indx = np.reshape(np.where(iter == arr_chanMap),(2,))
    if loc_indx[1] == 0:
        n_shankA += 1
    if loc_indx[1] == 1:
        n_shankB += 1
    if loc_indx[1] == 2:
        n_shankC += 1
    if loc_indx[1] == 3:
        n_shankD += 1
n_shank = np.array([n_shankA >= (n_density-7), n_shankB >= (n_density-7), n_shankC  >= (n_density-7), n_shankD  >= (n_density-7)])

# Filtering and reading data from Bin2Trials
A_missing = np.zeros((n_density,),dtype = np.single)
A_missing[:] = np.nan
B_missing = np.zeros((n_density,),dtype = np.single)
B_missing[:] = np.nan
C_missing = np.zeros((n_density,),dtype = np.single)
C_missing[:] = np.nan
D_missing = np.zeros((n_density,),dtype = np.single)
D_missing[:] = np.nan

print('Filtering and recovering data from .csv files --------------- \n')

for iter in chan_list:
    loc_indx = np.reshape(np.where(iter == arr_chanMap),(2,))
    if (loc_indx[1] == 0) & (n_shank[0] == True):
        A_missing[loc_indx[0]] = loc_indx[0]
        filename_str = 'Chan' + str(iter) + '.csv'
        filename = os.path.join(dir_Bin2Trials,filename_str)
        df_chan = pd.read_csv(filename, dtype = np.single)
        arr_chan = df_chan.to_numpy()
        data_eeg = arr_chan[:,:Ntrials]                     # Electrical recording data in microVolts
        data_eeg = data_eeg[n_stim_start-5*n_time_window:n_stim_start+5*n_time_window,:]
        data_eeg = filterSignal_BP_LFP(data_eeg, Fs,axis_value = 0)    # LFP b/w 0-160 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 60, axis_value = 0)     # Notch with cutoff 60 Hz
        # data_eeg = filterSignal_notch(data_eeg, Fs, 120, axis_value = 0)     # Notch with cutoff 120 Hz 
        Z_data_eeg = stats.zscore(data_eeg[n_time_window:-n_time_window,:], axis = 0)                         # Computing Z score (normalizing channels)
        # Z_data_eeg = data_eeg
        Adata_2D[loc_indx[0],:] = np.mean(Z_data_eeg[3*n_time_window:-(2*n_time_window),:], axis = 1)               # Averaging across trials
    elif (loc_indx[1] == 1) & (n_shank[1] == True):
        B_missing[loc_indx[0]] = loc_indx[0]
        filename_str = 'Chan' + str(iter) + '.csv'
        filename = os.path.join(dir_Bin2Trials,filename_str)
        df_chan = pd.read_csv(filename, dtype = np.single)
        arr_chan = df_chan.to_numpy()
        data_eeg = arr_chan[:,:Ntrials]         # Electrical recording data in microVolts
        data_eeg = data_eeg[n_stim_start-5*n_time_window:n_stim_start+5*n_time_window,:]
        data_eeg = filterSignal_BP_LFP(data_eeg, Fs,axis_value = 0)    # LFP b/w 0-160 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 60, axis_value = 0)     # Notch with cutoff 60 Hz
        # data_eeg = filterSignal_notch(data_eeg, Fs, 120, axis_value = 0)     # Notch with cutoff 120 Hz 
        Z_data_eeg = stats.zscore(data_eeg[n_time_window:-n_time_window,:], axis = 0)                         # Computing Z score (normalizing channels)
        # Z_data_eeg = data_eeg
        Bdata_2D[loc_indx[0],:] = np.mean(Z_data_eeg[3*n_time_window:-(2*n_time_window),:], axis = 1)               # Averaging across trials
    elif (loc_indx[1] == 2) & (n_shank[2] == True):
        C_missing[loc_indx[0]] = loc_indx[0]
        filename_str = 'Chan' + str(iter) + '.csv'
        filename = os.path.join(dir_Bin2Trials,filename_str)
        df_chan = pd.read_csv(filename, dtype = np.single)
        arr_chan = df_chan.to_numpy()
        data_eeg = arr_chan[:,:Ntrials]         # Electrical recording data in microVolts
        data_eeg = data_eeg[n_stim_start-5*n_time_window:n_stim_start+5*n_time_window,:]
        data_eeg = filterSignal_BP_LFP(data_eeg, Fs,axis_value = 0)    # LFP b/w 0-160 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 60, axis_value = 0)     # Notch with cutoff 60 Hz
        # data_eeg = filterSignal_notch(data_eeg, Fs, 120, axis_value = 0)     # Notch with cutoff 120 Hz 
        Z_data_eeg = stats.zscore(data_eeg[n_time_window:-n_time_window,:], axis = 0)                         # Computing Z score (normalizing channels)
        # Z_data_eeg = data_eeg
        Cdata_2D[loc_indx[0],:] = np.mean(Z_data_eeg[3*n_time_window:-(2*n_time_window),:], axis = 1)               # Averaging across trials
    elif (loc_indx[1] == 3) & (n_shank[3] == True):
        D_missing[loc_indx[0]] = loc_indx[0]
        filename_str = 'Chan' + str(iter) + '.csv'
        filename = os.path.join(dir_Bin2Trials,filename_str)
        df_chan = pd.read_csv(filename, dtype = np.single)
        arr_chan = df_chan.to_numpy()
        data_eeg = arr_chan[:,:Ntrials]         # Electrical recording data in microVolts
        data_eeg = data_eeg[n_stim_start-5*n_time_window:n_stim_start+5*n_time_window,:]
        data_eeg = filterSignal_BP_LFP(data_eeg, Fs,axis_value = 0)    # LFP b/w 0-160 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 60, axis_value = 0)     # Notch with cutoff 60 Hz
        # data_eeg = filterSignal_notch(data_eeg, Fs, 120, axis_value = 0)     # Notch with cutoff 120 Hz 
        Z_data_eeg = stats.zscore(data_eeg[n_time_window:-n_time_window,:], axis = 0)                         # Computing Z score (normalizing channels)
        # Z_data_eeg = data_eeg
        Ddata_2D[loc_indx[0],:] = np.mean(Z_data_eeg[3*n_time_window:-(2*n_time_window),:], axis = 1)               # Averaging across trials
        
    print('Chan ', iter, 'completed\n')

ACSD = np.zeros((n_density,3*n_time_window),dtype = np.float32)
BCSD = np.zeros((n_density,3*n_time_window),dtype = np.float32)
CCSD = np.zeros((n_density,3*n_time_window),dtype = np.float32)
DCSD = np.zeros((n_density,3*n_time_window),dtype = np.float32)

print('Computing CSD---------------------\n')

# Shank A
if n_shank[0] == True:
    data_in = np.transpose(Adata_2D)                           
    data_out_inter = 1e-3 * interp_chan_loss(data_in, A_missing)
    ACSD = CSD_compute(data_out_inter,Fs, spacing)
# Shank B
if n_shank[1] == True:
    data_in = np.transpose(Bdata_2D)
    data_out_inter = 1e-3 * interp_chan_loss(data_in, B_missing)
    BCSD = CSD_compute(data_out_inter,Fs, spacing)

# Shank C
if n_shank[2] == True:
    data_in = np.transpose(Cdata_2D)
    data_out_inter = 1e-3 * interp_chan_loss(data_in, C_missing)
    CCSD = CSD_compute(data_out_inter,Fs, spacing)

# Shank D
if n_shank[3] == True:
    data_in = np.transpose(Ddata_2D)
    data_out_inter = 1e-3 * interp_chan_loss(data_in, D_missing)
    DCSD = CSD_compute(data_out_inter,Fs, spacing)

# Plotting
fg3, axes = plt.subplots(1,4)
axes[0].set_title('Shank-A')
axes[1].set_title('Shank-B')
axes[2].set_title('Shank-C')
axes[3].set_title('Shank-D')
axes[0].set_ylabel('Peri-stimulation')
axes[0].set_xlabel('Time (ms)')
axes[1].set_xlabel('Time (ms)')
axes[2].set_xlabel('Time (ms)')
axes[3].set_xlabel('Time (ms)')
fg3.set_size_inches((18, 6), forward=True)
title_str = 'Peri-stimulation CSD- Avg of ' + str(Ntrials) + 'trials'
fg3.suptitle(title_str,fontweight = 'bold')



td_disp = 0.01
tt_disp = 0.04
M_mx = np.mean(ACSD) + 1.5*np.std(ACSD)
im = axes[0].imshow(ACSD[:,int(n_time_window-td_disp*Fs):int(n_time_window+tt_disp*Fs)], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [-td_disp*1000,tt_disp*1000,32,1],origin = 'upper', cmap = 'jet', interpolation = 'hamming')  
axes[0].vlines(0,1,32,linestyles = "dashed", colors = "k")
# M_mx = np.amax(np.abs(BCSD[:,:,0]))
M_mx = np.mean(BCSD) + 1.5*np.std(BCSD)
axes[1].imshow(BCSD[:,int(n_time_window-td_disp*Fs):int(n_time_window+tt_disp*Fs)], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [-td_disp*1000,tt_disp*1000,32,1],origin = 'upper', cmap = 'jet',interpolation = 'hamming')  
axes[1].vlines(0,1,32,linestyles = "dashed", colors = "k")
# M_mx = np.amax(np.abs(CCSD[:,:,0]))
M_mx = np.mean(CCSD) + 1.5*np.std(CCSD)
axes[2].imshow(CCSD[:,int(n_time_window-td_disp*Fs):int(n_time_window+tt_disp*Fs)], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [-td_disp*1000,tt_disp*1000,32,1],origin = 'upper', cmap = 'jet',interpolation = 'hamming')  
axes[2].vlines(0,1,32,linestyles = "dashed", colors = "k")
# M_mx = np.amax(np.abs(DCSD[:,:,0]))
M_mx = np.mean(DCSD) + 1.5*np.std(DCSD)
axes[3].imshow(DCSD[:,int(n_time_window-td_disp*Fs):int(n_time_window+tt_disp*Fs)], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [-td_disp*1000,tt_disp*1000,32,1],origin = 'upper', cmap = 'jet',interpolation = 'hamming')  
axes[3].vlines(0,1,32,linestyles = "dashed", colors = "k")
fg3.colorbar(im, ax = axes,label =  r'$\mu$'+'A/' + r'$mm^3$',location = 'bottom')
filename_save = 'Avg CSD of ' + str(Ntrials) + '.png'
filename_save = os.path.join(output_dir,filename_save)
plt.savefig(filename_save,format = 'png')
plt.close(fg3)
plt.clf()
plt.cla()

# The X axis of the plots need to be scaled properly



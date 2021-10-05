# -*- coding: utf-8 -*-
"""
Created on Sun July 19 00:20:16 2021
@author: Haad-Rathore
"""

from scipy.io import loadmat       # Import function to read data.
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from scipy import signal, integrate
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
dir_expsummary = os.path.join(dir_Bin2Trials,'exp_summary.xlsx')
if os.path.isdir(output_dir):
	os.rmdir(output_dir)        # delete directory if it already exists
os.makedirs(output_dir)


# Extracting data from summary file .xlsx
df_exp_summary = pd.read_excel(dir_expsummary)
arr_exp_summary = df_exp_summary.to_numpy()
Num_chan = arr_exp_summary[0,0]         # Number of channels
Notch_freq = arr_exp_summary[0,1]       # Notch frequencey selected (in Hz)
Fs = arr_exp_summary[0,2]               # Sampling freq (in Hz)
stim_start_time = arr_exp_summary[2,2]  # Stimulation start
n_stim_start = int(Fs * stim_start_time)# Stimulation start time in samples
Ntrials = arr_exp_summary[2,4]          # Number of trials
stim_end_time = arr_exp_summary[2,1] + stim_start_time  # End time of stimulation
time_seq = arr_exp_summary[2,0]         # Time of one sequence in seconds
Seq_perTrial =  arr_exp_summary[2,3]    # Number of sequences per trial
total_time = time_seq * Seq_perTrial    # Total time of the trial
print('Each sequence is: ', time_seq, 'sec')
time_seq = int(np.ceil(time_seq * Fs/2) * 2)                # Time of one sequence in samples (rounded up to even)

# --------------------- SET THESE PARAMETERS ------------------------------
time_window = 200e-3                    # Selecting a 200 ms time window
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
source_dir_list = natsorted(os.listdir(dir_Bin2Trials))


# First channel only 
filename = os.path.join(dir_Bin2Trials,'Chan0.csv')
df_chan = pd.read_csv(filename,dtype = np.single)
arr_chan = df_chan.to_numpy()
t = arr_chan[:,-1]
adc_data = arr_chan[:,-2]

# Creating 3D matrices to store the data
Adata3D_pre = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Adata3D_pre[:] = np.nan
Adata3D_stim = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Adata3D_stim[:] = np.nan
Adata3D_post = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Adata3D_post[:] = np.nan
Bdata3D_pre = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Bdata3D_pre[:] = np.nan
Bdata3D_stim = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Bdata3D_stim[:] = np.nan
Bdata3D_post = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Bdata3D_post[:] = np.nan
Cdata3D_pre = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Cdata3D_pre[:] = np.nan
Cdata3D_stim = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Cdata3D_stim[:] = np.nan
Cdata3D_post = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Cdata3D_post[:] = np.nan
Ddata3D_pre = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Ddata3D_pre[:] = np.nan
Ddata3D_stim = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Ddata3D_stim[:] = np.nan
Ddata3D_post = np.zeros((arr_chanMap.shape[0],n_time_window,Ntrials),dtype = np.single)
Ddata3D_post[:] = np.nan

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
        data_eeg = arr_chan[:,:Ntrials]         # Electrical recording data in microVolts
        data_eeg = data_eeg[0:n_stim_start+3*n_time_window,:]
        data_eeg = np.transpose(data_eeg)
        data_eeg = filterSignal_lowpassLFP(data_eeg, Fs)    # LFP b/w 0-150 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 60)     # Notch with cutoff 60 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 120)     # Notch with cutoff 120 Hz 
        data_eeg = np.transpose(data_eeg)
        Adata3D_pre[loc_indx[0],:,:] = data_eeg[n_stim_start-n_time_window+1:n_stim_start+1,:]
        Adata3D_stim[loc_indx[0],:,:] = data_eeg[n_stim_start+1:n_stim_start+n_time_window+1,:]
        Adata3D_post[loc_indx[0],:,:] = data_eeg[n_stim_start+n_time_window+1:n_stim_start+2*n_time_window+1,:]
    elif (loc_indx[1] == 1) & (n_shank[1] == True):
        B_missing[loc_indx[0]] = loc_indx[0]
        filename_str = 'Chan' + str(iter) + '.csv'
        filename = os.path.join(dir_Bin2Trials,filename_str)
        df_chan = pd.read_csv(filename, dtype = np.single)
        arr_chan = df_chan.to_numpy()
        data_eeg = arr_chan[:,:Ntrials]         # Electrical recording data in microVolts
        data_eeg = data_eeg[0:n_stim_start+3*n_time_window,:]
        data_eeg = np.transpose(data_eeg)
        data_eeg = filterSignal_lowpassLFP(data_eeg, Fs)    # LFP b/w 0-150 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 60)     # Notch with cutoff 60 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 120)     # Notch with cutoff 120 Hz 
        data_eeg = np.transpose(data_eeg)
        Bdata3D_pre[loc_indx[0],:,:] = data_eeg[n_stim_start-n_time_window+1:n_stim_start+1,:]
        Bdata3D_stim[loc_indx[0],:,:] = data_eeg[n_stim_start+1:n_stim_start+n_time_window+1,:]
        Bdata3D_post[loc_indx[0],:,:] = data_eeg[n_stim_start+n_time_window+1:n_stim_start+2*n_time_window+1,:]
    elif (loc_indx[1] == 2) & (n_shank[2] == True):
        C_missing[loc_indx[0]] = loc_indx[0]
        filename_str = 'Chan' + str(iter) + '.csv'
        filename = os.path.join(dir_Bin2Trials,filename_str)
        df_chan = pd.read_csv(filename, dtype = np.single)
        arr_chan = df_chan.to_numpy()
        data_eeg = arr_chan[:,:Ntrials]         # Electrical recording data in microVolts
        data_eeg = data_eeg[0:n_stim_start+3*n_time_window,:]
        data_eeg = np.transpose(data_eeg)
        data_eeg = filterSignal_lowpassLFP(data_eeg, Fs)    # LFP b/w 0-150 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 60)     # Notch with cutoff 60 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 120)     # Notch with cutoff 120 Hz 
        data_eeg = np.transpose(data_eeg)
        Cdata3D_pre[loc_indx[0],:,:] = data_eeg[n_stim_start-n_time_window+1:n_stim_start+1,:]
        Cdata3D_stim[loc_indx[0],:,:] = data_eeg[n_stim_start+1:n_stim_start+n_time_window+1,:]
        Cdata3D_post[loc_indx[0],:,:] = data_eeg[n_stim_start+n_time_window+1:n_stim_start+2*n_time_window+1,:]
    elif (loc_indx[1] == 3) & (n_shank[3] == True):
        D_missing[loc_indx[0]] = loc_indx[0]
        filename_str = 'Chan' + str(iter) + '.csv'
        filename = os.path.join(dir_Bin2Trials,filename_str)
        df_chan = pd.read_csv(filename, dtype = np.single)
        arr_chan = df_chan.to_numpy()
        data_eeg = arr_chan[:,:Ntrials]         # Electrical recording data in microVolts
        data_eeg = data_eeg[0:n_stim_start+3*n_time_window,:]
        data_eeg = np.transpose(data_eeg)
        data_eeg = filterSignal_lowpassLFP(data_eeg, Fs)    # LFP b/w 0-150 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 60)     # Notch with cutoff 60 Hz
        data_eeg = filterSignal_notch(data_eeg, Fs, 120)     # Notch with cutoff 120 Hz 
        data_eeg = np.transpose(data_eeg)
        Ddata3D_pre[loc_indx[0],:,:] = data_eeg[n_stim_start-n_time_window+1:n_stim_start+1,:]
        Ddata3D_stim[loc_indx[0],:,:] = data_eeg[n_stim_start+1:n_stim_start+n_time_window+1,:]
        Ddata3D_post[loc_indx[0],:,:] = data_eeg[n_stim_start+n_time_window+1:n_stim_start+2*n_time_window+1,:]
        
    print((iter+1)/Num_chan*100,'% done')

ACSD = np.zeros((n_density,n_time_window,3),dtype = np.float32)
BCSD = np.zeros((n_density,n_time_window,3),dtype = np.float32)
CCSD = np.zeros((n_density,n_time_window,3),dtype = np.float32)
DCSD = np.zeros((n_density,n_time_window,3),dtype = np.float32)
mean_ACSD = np.zeros((n_density,n_time_window,3),dtype = np.float32)        # for trial averaging
mean_BCSD = np.zeros((n_density,n_time_window,3),dtype = np.float32)        # for trial averaging
mean_CCSD = np.zeros((n_density,n_time_window,3),dtype = np.float32)        # for trial averaging
mean_DCSD = np.zeros((n_density,n_time_window,3),dtype = np.float32)        # for trial averaging

print('Computing CSD---------------------\n')

for iter in range(0,Ntrials): 
    # Shank A
    if n_shank[0] == True:
        data_in = np.transpose(Adata3D_pre[:,:,iter])                           
        data_out_inter = 1e-6 * interp_chan_loss(data_in, A_missing)
        ACSD[:,:,0] = CSD_compute(data_out_inter,Fs, spacing)
        mean_ACSD[:,:,0] = mean_ACSD[:,:,0] + ACSD[:,:,0]
        data_in = np.transpose(Adata3D_stim[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, A_missing)
        ACSD[:,:,1] = CSD_compute(data_out_inter,Fs, spacing)
        mean_ACSD[:,:,1] = mean_ACSD[:,:,1] + ACSD[:,:,1]
        data_in = np.transpose(Adata3D_post[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, A_missing)
        ACSD[:,:,2] = CSD_compute(data_out_inter,Fs, spacing)
        mean_ACSD[:,:,2] = mean_ACSD[:,:,2] + ACSD[:,:,2]
    # Shank B
    if n_shank[1] == True:
        data_in = np.transpose(Bdata3D_pre[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, B_missing)
        BCSD[:,:,0] = CSD_compute(data_out_inter,Fs, spacing)
        mean_BCSD[:,:,0] = mean_BCSD[:,:,0] + BCSD[:,:,0]
        data_in = np.transpose(Bdata3D_stim[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, B_missing)
        BCSD[:,:,1] = CSD_compute(data_out_inter,Fs, spacing)
        mean_BCSD[:,:,1] = mean_BCSD[:,:,1] + BCSD[:,:,1]
        data_in = np.transpose(Bdata3D_post[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, B_missing)
        BCSD[:,:,2] = CSD_compute(data_out_inter,Fs, spacing)
        mean_BCSD[:,:,2] = mean_BCSD[:,:,2] + BCSD[:,:,2]
    # Shank C
    if n_shank[2] == True:
        data_in = np.transpose(Cdata3D_pre[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, C_missing)
        CCSD[:,:,0] = CSD_compute(data_out_inter,Fs, spacing)
        mean_CCSD[:,:,0] = mean_CCSD[:,:,0] + CCSD[:,:,0]
        data_in = np.transpose(Cdata3D_stim[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, C_missing)
        CCSD[:,:,1] = CSD_compute(data_out_inter,Fs, spacing)
        mean_CCSD[:,:,1] = mean_CCSD[:,:,1] + CCSD[:,:,1]
        data_in = np.transpose(Cdata3D_post[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, C_missing)
        CCSD[:,:,2] = CSD_compute(data_out_inter,Fs, spacing)
        mean_CCSD[:,:,2] = mean_CCSD[:,:,2] + CCSD[:,:,2]
    # Shank D
    if n_shank[3] == True:
        data_in = np.transpose(Ddata3D_pre[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, D_missing)
        DCSD[:,:,0] = CSD_compute(data_out_inter,Fs, spacing)
        mean_DCSD[:,:,0] = mean_DCSD[:,:,0] + DCSD[:,:,0]
        data_in = np.transpose(Ddata3D_stim[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, D_missing)
        DCSD[:,:,1] = CSD_compute(data_out_inter,Fs, spacing)
        mean_DCSD[:,:,1] = mean_DCSD[:,:,1] + DCSD[:,:,1]
        data_in = np.transpose(Ddata3D_post[:,:,iter])
        data_out_inter = 1e-6 * interp_chan_loss(data_in, D_missing)
        DCSD[:,:,2] = CSD_compute(data_out_inter,Fs, spacing)
        mean_DCSD[:,:,2] = mean_DCSD[:,:,2] + DCSD[:,:,2]

    # Plotting
    fg2, axes = plt.subplots(3,4)
    axes[0,0].set_title('Shank-A')
    axes[0,1].set_title('Shank-B')
    axes[0,2].set_title('Shank-C')
    axes[0,3].set_title('Shank-D')
    axes[0,0].set_ylabel('Pre-stimulation')
    axes[1,0].set_ylabel('Stimulation')
    axes[2,0].set_ylabel('Post-stimulation')
    axes[2,0].set_xlabel('Time (ms)')
    axes[2,1].set_xlabel('Time (ms)')
    axes[2,2].set_xlabel('Time (ms)')
    axes[2,3].set_xlabel('Time (ms)')
    axes[0,1].get_xaxis().set_visible(False);axes[0,1].get_yaxis().set_visible(False)
    axes[0,2].get_xaxis().set_visible(False);axes[0,2].get_yaxis().set_visible(False)
    axes[0,3].get_xaxis().set_visible(False);axes[0,3].get_yaxis().set_visible(False)
    axes[1,1].get_xaxis().set_visible(False);axes[1,1].get_yaxis().set_visible(False)
    axes[1,2].get_xaxis().set_visible(False);axes[1,2].get_yaxis().set_visible(False)
    axes[1,3].get_xaxis().set_visible(False);axes[1,3].get_yaxis().set_visible(False)
    axes[0,0].get_xaxis().set_visible(False)
    axes[1,0].get_xaxis().set_visible(False)
    axes[2,1].get_yaxis().set_visible(False)
    axes[2,2].get_yaxis().set_visible(False)
    axes[2,3].get_yaxis().set_visible(False)
    fg2.set_size_inches((10, 13), forward=True)
    title_str = 'Peri-stimulation CSD Trial- ' + str(iter+1)
    fg2.suptitle(title_str,fontweight = 'bold')
    
    
    # M_mx = np.amax(np.abs(ACSD[:,:,0]))
    M_mx = np.mean(ACSD[:,:,1]) + np.std(ACSD[:,:,1])
    im = axes[0,0].imshow(ACSD[:,:,0], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(ACSD[:,:,1]))
    axes[1,0].imshow(ACSD[:,:,1], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(ACSD[:,:,2]))
    axes[2,0].imshow(ACSD[:,:,2], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(BCSD[:,:,0]))
    M_mx = np.mean(BCSD[:,:,1]) + np.std(BCSD[:,:,1])
    axes[0,1].imshow(BCSD[:,:,0], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(BCSD[:,:,1]))
    axes[1,1].imshow(BCSD[:,:,1], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(BCSD[:,:,2]))
    axes[2,1].imshow(BCSD[:,:,2], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(CCSD[:,:,0]))
    M_mx = np.mean(CCSD[:,:,1]) + np.std(CCSD[:,:,1])
    axes[0,2].imshow(CCSD[:,:,0], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(CCSD[:,:,1]))
    axes[1,2].imshow(CCSD[:,:,1], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(CCSD[:,:,2]))
    axes[2,2].imshow(CCSD[:,:,2], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(DCSD[:,:,0]))
    M_mx = np.mean(DCSD[:,:,1]) + np.std(DCSD[:,:,1])
    axes[0,3].imshow(DCSD[:,:,0], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(DCSD[:,:,1]))
    axes[1,3].imshow(DCSD[:,:,1], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    # M_mx = np.amax(np.abs(DCSD[:,:,2]))
    axes[2,3].imshow(DCSD[:,:,2], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
    
    
    
    fg2.colorbar(im, ax = axes,label =  r'$\mu$'+'A/' + r'$mm^3$',location = 'right')
    filename_save = 'Trial-' + str(iter+1) + '.png'
    filename_save = os.path.join(output_dir,filename_save)
    plt.savefig(filename_save,format = 'png')
    plt.close(fg2)
    plt.clf()
    plt.cla()

# Averaged plots over all trials
mean_ACSD = mean_ACSD/(n_density-1)
mean_BCSD = mean_BCSD/(n_density-1)
mean_CCSD = mean_CCSD/(n_density-1)
mean_DCSD = mean_DCSD/(n_density-1)

# Plotting
fg3, axes = plt.subplots(3,4)
axes[0,0].set_title('Shank-A')
axes[0,1].set_title('Shank-B')
axes[0,2].set_title('Shank-C')
axes[0,3].set_title('Shank-D')
axes[0,0].set_ylabel('Pre-stimulation')
axes[1,0].set_ylabel('Stimulation')
axes[2,0].set_ylabel('Post-stimulation')
axes[2,0].set_xlabel('Time (ms)')
axes[2,1].set_xlabel('Time (ms)')
axes[2,2].set_xlabel('Time (ms)')
axes[2,3].set_xlabel('Time (ms)')
axes[0,1].get_xaxis().set_visible(False);axes[0,1].get_yaxis().set_visible(False)
axes[0,2].get_xaxis().set_visible(False);axes[0,2].get_yaxis().set_visible(False)
axes[0,3].get_xaxis().set_visible(False);axes[0,3].get_yaxis().set_visible(False)
axes[1,1].get_xaxis().set_visible(False);axes[1,1].get_yaxis().set_visible(False)
axes[1,2].get_xaxis().set_visible(False);axes[1,2].get_yaxis().set_visible(False)
axes[1,3].get_xaxis().set_visible(False);axes[1,3].get_yaxis().set_visible(False)
axes[0,0].get_xaxis().set_visible(False)
axes[1,0].get_xaxis().set_visible(False)
axes[2,1].get_yaxis().set_visible(False)
axes[2,2].get_yaxis().set_visible(False)
axes[2,3].get_yaxis().set_visible(False)
fg3.set_size_inches((10, 13), forward=True)
title_str = 'Peri-stimulation CSD- Avg of ' + str(Ntrials) + 'trials'
fg3.suptitle(title_str,fontweight = 'bold')


# M_mx = np.amax(np.abs(ACSD[:,:,0]))
M_mx = np.mean(mean_ACSD[:,:,1]) + np.std(mean_ACSD[:,:,1])
im = axes[0,0].imshow(mean_ACSD[:,:,0], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
axes[1,0].imshow(mean_ACSD[:,:,1], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
axes[2,0].imshow(mean_ACSD[:,:,2], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
# M_mx = np.amax(np.abs(BCSD[:,:,0]))
M_mx = np.mean(mean_BCSD[:,:,1]) + np.std(mean_BCSD[:,:,1])
axes[0,1].imshow(mean_BCSD[:,:,0], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
axes[1,1].imshow(mean_BCSD[:,:,1], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
axes[2,1].imshow(mean_BCSD[:,:,2], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
# M_mx = np.amax(np.abs(CCSD[:,:,0]))
M_mx = np.mean(mean_CCSD[:,:,1]) + np.std(mean_CCSD[:,:,1])
axes[0,2].imshow(mean_CCSD[:,:,0], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
axes[1,2].imshow(mean_CCSD[:,:,1], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
axes[2,2].imshow(mean_CCSD[:,:,2], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
# M_mx = np.amax(np.abs(DCSD[:,:,0]))
M_mx = np.mean(mean_DCSD[:,:,1]) + np.std(mean_DCSD[:,:,1])
axes[0,3].imshow(mean_DCSD[:,:,0], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
axes[1,3].imshow(mean_DCSD[:,:,1], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  
axes[2,3].imshow(mean_DCSD[:,:,2], vmax = M_mx, vmin = -M_mx, aspect = 'auto', extent = [0,200,32,1],origin = 'upper', cmap = 'jet_r')  


fg3.colorbar(im, ax = axes,label =  r'$\mu$'+'A/' + r'$mm^3$',location = 'right')
filename_save = 'Avg CSD of ' + str(Ntrials) + '.png'
filename_save = os.path.join(output_dir,filename_save)
plt.savefig(filename_save,format = 'png')
plt.close(fg3)
plt.clf()
plt.cla()





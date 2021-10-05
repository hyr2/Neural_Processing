# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:07:16 2021

@author: Haad-Rathore
"""

from scipy.io import loadmat, savemat      # Import function to read data
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import colors as cl
import sys, os, shutil
import numpy as np
from natsort import natsorted
import pandas as pd
from Support import *

source_dir = input('Enter the source directory: \n')
dir_mat = input('Enter the spectrogram_mat folder path: \n')
output_dir = os.path.join(source_dir,'Processed','depth-spectrogram')
dir_chan_list = os.path.join(source_dir,'chan_list.xlsx')
dir_chan_map = os.path.join(source_dir,'chan_map_1x32_128ch.xlsx')
dir_expsummary = os.path.join(source_dir,'exp_summary.xlsx')
dir_mat_list = natsorted(os.listdir(dir_mat))
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

# Channel maps
df_chanMap = pd.read_excel(dir_chan_map,usecols=[0,1,2,3],header = None,sheet_name = 2)
arr_chanMap = df_chanMap.to_numpy()                 # 4 shank device 1x32 channels on each shank
df_chanList = pd.read_excel(dir_chan_list,header = 0)
chan_map = df_chanList.to_numpy()
chan_map = np.reshape(chan_map,(Num_chan,))

# Set these parameters
electrode_spacing = 25
n_density = 32
start_depth = 0

# Cortical depth arrays (Short-time FT)
filename_mat = dir_mat_list[0]
filename_mat = os.path.join(dir_mat,filename_mat)
dict_electrical = loadmat(filename_mat)
t_electrical = dict_electrical['time']
f_LFP = dict_electrical['f_LFP']
f_LFP = np.reshape(f_LFP,(f_LFP.shape[1],))
t_electrical = np.reshape(t_electrical,(t_electrical.shape[1],))
delta_t = t_electrical[1] - t_electrical[0]
t_activation = np.where(np.logical_and(t_electrical>=stim_start_time, t_electrical<=stim_end_time))
t_activation = np.asarray(t_activation)
t_activation = np.reshape(t_activation,(t_activation.size,))

# Cortical depth arrays (Morlet Transform)
t_cwt = dict_electrical['t_cwt']
f_cwt = dict_electrical['f_cwt']
f_cwt = np.reshape(f_cwt,(f_cwt.shape[1],))
t_cwt = np.reshape(t_cwt,(t_cwt.shape[1],))
delta_t_cwt = t_cwt[1] - t_cwt[0]
t_activation_cwt = np.where(np.logical_and(t_cwt>=stim_start_time, t_cwt<=stim_end_time))
t_activation_cwt = np.asarray(t_activation_cwt)
t_activation_cwt = np.reshape(t_activation_cwt,(t_activation_cwt.size,))

# Morlet_CWT = dict_electrical['Morlet_CWT']


# Depth
LFP_depth_mean = np.zeros((arr_chanMap.shape[0],arr_chanMap.shape[1],len(f_LFP)))
LFP_depth_mean_cwt = np.zeros((arr_chanMap.shape[0],arr_chanMap.shape[1],len(f_cwt)))
LFP_depth_mean[:] = np.nan
depth_shank = np.arange(start_depth,start_depth + electrode_spacing*(arr_chanMap.shape[0]),2*electrode_spacing)
A_missing = np.zeros((n_density,),dtype = np.single)
A_missing[:] = np.nan
B_missing = np.zeros((n_density,),dtype = np.single)
B_missing[:] = np.nan
C_missing = np.zeros((n_density,),dtype = np.single)
C_missing[:] = np.nan
D_missing = np.zeros((n_density,),dtype = np.single)
D_missing[:] = np.nan




# Loading electrical data Shank A
# finding location of this channel from the channel map
for iter_chan in chan_map:
	chan_loc = np.where(arr_chanMap == iter_chan)
	chan_loc = np.asarray(chan_loc)
	chan_loc = np.reshape(chan_loc,(2,))        # first element is depth index and second element is shank index
	
	if chan_loc[1] == 0:
		A_missing[chan_loc[0]] = chan_loc[0]
		filename_mat = 'Chan' + str(iter_chan) + '.mat'
		filename_mat = os.path.join(dir_mat,filename_mat)
		dict_electrical = loadmat(filename_mat)
		
		spectrogram_ndPSD = dict_electrical['LFP_ndPSD']
		spectrogram_ndPSD = np.reshape(spectrogram_ndPSD,(len(f_LFP),len(t_electrical)))
		morlet = dict_electrical['Morlet_CWT']
		morlet = np.reshape(morlet,(len(f_cwt),len(t_cwt)))
		LFP_depth_mean[chan_loc[0],chan_loc[1],:] = np.mean(spectrogram_ndPSD[:,t_activation[0:]],axis = 1)
		LFP_depth_mean_cwt[chan_loc[0],chan_loc[1],:] = np.mean(morlet[:,t_activation_cwt[0:]], axis = 1)
		
	if chan_loc[1] == 1:
		B_missing[chan_loc[0]] = chan_loc[0]
		filename_mat = 'Chan' + str(iter_chan) + '.mat'
		filename_mat = os.path.join(dir_mat,filename_mat)
		
		dict_electrical = loadmat(filename_mat)
		spectrogram_ndPSD = dict_electrical['LFP_ndPSD']
		spectrogram_ndPSD = np.reshape(spectrogram_ndPSD,(len(f_LFP),len(t_electrical)))
		morlet = dict_electrical['Morlet_CWT']
		morlet = np.reshape(morlet,(len(f_cwt),len(t_cwt)))
		LFP_depth_mean[chan_loc[0],chan_loc[1],:] = np.mean(spectrogram_ndPSD[:,t_activation[0:]],axis = 1)
		LFP_depth_mean_cwt[chan_loc[0],chan_loc[1],:] = np.mean(morlet[:,t_activation_cwt[0:]], axis = 1)
		
	if chan_loc[1] == 2:
		C_missing[chan_loc[0]] = chan_loc[0]
		filename_mat = 'Chan' + str(iter_chan) + '.mat'
		filename_mat = os.path.join(dir_mat,filename_mat)
		
		dict_electrical = loadmat(filename_mat)
		spectrogram_ndPSD = dict_electrical['LFP_ndPSD']
		spectrogram_ndPSD = np.reshape(spectrogram_ndPSD,(len(f_LFP),len(t_electrical)))
		morlet = dict_electrical['Morlet_CWT']
		morlet = np.reshape(morlet,(len(f_cwt),len(t_cwt)))
		LFP_depth_mean[chan_loc[0],chan_loc[1],:] = np.mean(spectrogram_ndPSD[:,t_activation[0:]],axis = 1)
		LFP_depth_mean_cwt[chan_loc[0],chan_loc[1],:] = np.mean(morlet[:,t_activation_cwt[0:]], axis = 1)
	
	if chan_loc[1] == 3:
		D_missing[chan_loc[0]] = chan_loc[0]
		filename_mat = 'Chan' + str(iter_chan) + '.mat'
		filename_mat = os.path.join(dir_mat,filename_mat)
		
		dict_electrical = loadmat(filename_mat)
		spectrogram_ndPSD = dict_electrical['LFP_ndPSD']
		spectrogram_ndPSD = np.reshape(spectrogram_ndPSD,(len(f_LFP),len(t_electrical)))
		morlet = dict_electrical['Morlet_CWT']
		morlet = np.reshape(morlet,(len(f_cwt),len(t_cwt)))
		LFP_depth_mean[chan_loc[0],chan_loc[1],:] = np.mean(spectrogram_ndPSD[:,t_activation[0:]],axis = 1)
		LFP_depth_mean_cwt[chan_loc[0],chan_loc[1],:] = np.mean(morlet[:,t_activation_cwt[0:]], axis = 1)
		

# A_missing[16] = np.nan
# LFP_depth_mean[16,0,:] = np.nan

data_out_A = interp_chan_loss(np.transpose(LFP_depth_mean[:,0,:]),A_missing)
data_out_B = interp_chan_loss(np.transpose(LFP_depth_mean[:,1,:]),B_missing)
data_out_C = interp_chan_loss(np.transpose(LFP_depth_mean[:,2,:]),C_missing)
data_out_D = interp_chan_loss(np.transpose(LFP_depth_mean[:,3,:]),D_missing)

data_out_A_cwt = interp_chan_loss(np.transpose(LFP_depth_mean_cwt[:,0,:]),A_missing)
data_out_B_cwt = interp_chan_loss(np.transpose(LFP_depth_mean_cwt[:,1,:]),B_missing)
data_out_C_cwt = interp_chan_loss(np.transpose(LFP_depth_mean_cwt[:,2,:]),C_missing)
data_out_D_cwt = interp_chan_loss(np.transpose(LFP_depth_mean_cwt[:,3,:]),D_missing)

# plotting short time FT analysis plots
fig, ax = plt.subplots(1,1)
filename_save_svg = 'cortical-depth_A_STFT.svg'
filename_save_png = 'cortical-depth_A_STFT.png'
im = ax.imshow(data_out_A,interpolation = 'spline16',aspect = 'auto', extent = [depth_shank[0],depth_shank[-1],f_LFP[0],f_LFP[-1]], origin = 'lower')
fig.colorbar(im, ax = ax,label = r'$\Delta$'+r'$P_n$')
ax.set_ylabel('Freq (Hz)', fontsize = 16)
ax.set_xlabel('Cortical depth (um)', fontsize = 16)
fig.set_size_inches((8, 6), forward=True)
fig.tight_layout() 
# plt.savefig(os.path.join(folder_chanmap,filename_save_svg),format = 'svg')
plt.savefig(os.path.join(output_dir,filename_save_png),format = 'png')
plt.close(fig)
plt.clf()
plt.cla()

fig, ax = plt.subplots(1,1)
filename_save_svg = 'cortical-depth_B.svg'
filename_save_png = 'cortical-depth_B_STFT.png'
im = ax.imshow(data_out_B,interpolation = 'spline16',aspect = 'auto', extent = [depth_shank[0],depth_shank[-1],f_LFP[0],f_LFP[-1]], origin = 'lower')
fig.colorbar(im, ax = ax,label = r'$\Delta$'+r'$P_n$')
ax.set_ylabel('Freq (Hz)', fontsize = 16)
ax.set_xlabel('Cortical depth (um)', fontsize = 16)
fig.set_size_inches((8, 6), forward=True)
fig.tight_layout() 
# plt.savefig(os.path.join(folder_chanmap,filename_save_svg),format = 'svg')
plt.savefig(os.path.join(output_dir,filename_save_png),format = 'png')
plt.close(fig)
plt.clf()
plt.cla()

fig, ax = plt.subplots(1,1)
filename_save_svg = 'cortical-depth_C.svg'
filename_save_png = 'cortical-depth_C_STFT.png'
im = ax.imshow(data_out_C,interpolation = 'spline16',aspect = 'auto', extent = [depth_shank[0],depth_shank[-1],f_LFP[0],f_LFP[-1]], origin = 'lower')
fig.colorbar(im, ax = ax,label = r'$\Delta$'+r'$P_n$')
ax.set_ylabel('Freq (Hz)', fontsize = 16)
ax.set_xlabel('Cortical depth (um)', fontsize = 16)
fig.set_size_inches((8, 6), forward=True)
fig.tight_layout() 
# plt.savefig(os.path.join(folder_chanmap,filename_save_svg),format = 'svg')
plt.savefig(os.path.join(output_dir,filename_save_png),format = 'png')
plt.close(fig)
plt.clf()
plt.cla()

fig, ax = plt.subplots(1,1)
filename_save_svg = 'cortical-depth_D.svg'
filename_save_png = 'cortical-depth_D_STFT.png'
im = ax.imshow(data_out_D,interpolation = 'spline16',aspect = 'auto', extent = [depth_shank[0],depth_shank[-1],f_LFP[0],f_LFP[-1]], origin = 'lower')
fig.colorbar(im, ax = ax,label = r'$\Delta$'+r'$P_n$')
ax.set_ylabel('Freq (Hz)', fontsize = 16)
ax.set_xlabel('Cortical depth (um)', fontsize = 16)
fig.set_size_inches((8, 6), forward=True)
fig.tight_layout() 
# plt.savefig(os.path.join(folder_chanmap,filename_save_svg),format = 'svg')
plt.savefig(os.path.join(output_dir,filename_save_png),format = 'png')
plt.close(fig)
plt.clf()
plt.cla()

# plotting morlet analysis plots
LL = len(f_cwt)
y_label_list = np.around([f_cwt[0],f_cwt[int(LL/4)],f_cwt[int(LL/2)],f_cwt[int(3/4*LL)],f_cwt[-1]])
# y_label_list = np.flip(y_label_list)


fig, ax = plt.subplots(1,1)
filename_save_svg = 'cortical-depth_A_STFT.svg'
filename_save_png = 'cortical-depth_A_cwt.png'
max_lim = np.mean(data_out_A_cwt) + 1.5*np.std(data_out_A_cwt)
min_lim = np.mean(data_out_A_cwt) - 1.5*np.std(data_out_A_cwt)
im = ax.imshow(data_out_A_cwt,interpolation = 'spline16',aspect = 'auto', origin = 'upper', vmax = max_lim, vmin = min_lim)
fig.colorbar(im, ax = ax,label = r'$\Delta$'+r'$P_n$')
ax.set_ylabel('Freq (Hz)', fontsize = 16)
ax.set_xlabel('Cortical depth (um)', fontsize = 16)
ax.set_yticks([0,int(LL/4),int(LL/2),int(3/4*LL),LL-1])
ax.set_yticklabels(y_label_list)
ax.set_xticks(np.arange(0,32,2))
ax.set_xticklabels(depth_shank)
fig.set_size_inches((8, 6), forward=True)
fig.tight_layout() 
# plt.savefig(os.path.join(folder_chanmap,filename_save_svg),format = 'svg')
plt.savefig(os.path.join(output_dir,filename_save_png),format = 'png')
plt.close(fig)
plt.clf()
plt.cla()

fig, ax = plt.subplots(1,1)
filename_save_svg = 'cortical-depth_B.svg'
filename_save_png = 'cortical-depth_B_cwt.png'
max_lim = np.mean(data_out_B_cwt) + 1.5*np.std(data_out_B_cwt)
min_lim = np.mean(data_out_B_cwt) - 1.5*np.std(data_out_B_cwt)
im = ax.imshow(data_out_B_cwt,interpolation = 'spline16',aspect = 'auto', origin = 'upper', vmax = max_lim, vmin = min_lim)
fig.colorbar(im, ax = ax,label = r'$\Delta$'+r'$P_n$')
ax.set_ylabel('Freq (Hz)', fontsize = 16)
ax.set_xlabel('Cortical depth (um)', fontsize = 16)
ax.set_yticks([0,int(LL/4),int(LL/2),int(3/4*LL),LL-1])
ax.set_yticklabels(y_label_list)
ax.set_xticks(np.arange(0,32,2))
ax.set_xticklabels(depth_shank)
fig.set_size_inches((8, 6), forward=True)
fig.tight_layout() 
# plt.savefig(os.path.join(folder_chanmap,filename_save_svg),format = 'svg')
plt.savefig(os.path.join(output_dir,filename_save_png),format = 'png')
plt.close(fig)
plt.clf()
plt.cla()

fig, ax = plt.subplots(1,1)
filename_save_svg = 'cortical-depth_C.svg'
filename_save_png = 'cortical-depth_C_cwt.png'
max_lim = np.mean(data_out_C_cwt) + 1.5*np.std(data_out_C_cwt)
min_lim = np.mean(data_out_C_cwt) - 1.5*np.std(data_out_C_cwt)
im = ax.imshow(data_out_C_cwt,interpolation = 'spline16',aspect = 'auto', origin = 'upper', vmax = max_lim, vmin = min_lim)
fig.colorbar(im, ax = ax,label = r'$\Delta$'+r'$P_n$')
ax.set_ylabel('Freq (Hz)', fontsize = 16)
ax.set_xlabel('Cortical depth (um)', fontsize = 16)
ax.set_yticks([0,int(LL/4),int(LL/2),int(3/4*LL),LL-1])
ax.set_yticklabels(y_label_list)
ax.set_xticks(np.arange(0,32,2))
ax.set_xticklabels(depth_shank)
fig.set_size_inches((8, 6), forward=True)
fig.tight_layout() 
# plt.savefig(os.path.join(folder_chanmap,filename_save_svg),format = 'svg')
plt.savefig(os.path.join(output_dir,filename_save_png),format = 'png')
plt.close(fig)
plt.clf()
plt.cla()

fig, ax = plt.subplots(1,1)
filename_save_svg = 'cortical-depth_D.svg'
filename_save_png = 'cortical-depth_D_cwt.png'
max_lim = np.mean(data_out_D_cwt) + 1.5*np.std(data_out_D_cwt)
min_lim = np.mean(data_out_D_cwt) - 1.5*np.std(data_out_D_cwt)
im = ax.imshow(data_out_D_cwt,interpolation = 'spline16',aspect = 'auto', origin = 'upper', vmax = max_lim, vmin = min_lim)
fig.colorbar(im, ax = ax,label = r'$\Delta$'+r'$P_n$')
ax.set_ylabel('Freq (Hz)', fontsize = 16)
ax.set_xlabel('Cortical depth (um)', fontsize = 16)
ax.set_yticks([0,int(LL/4),int(LL/2),int(3/4*LL),LL-1])
ax.set_yticklabels(y_label_list)
ax.set_xticks(np.arange(0,32,2))
ax.set_xticklabels(depth_shank)
fig.set_size_inches((8, 6), forward=True)
fig.tight_layout() 
# plt.savefig(os.path.join(folder_chanmap,filename_save_svg),format = 'svg')
plt.savefig(os.path.join(output_dir,filename_save_png),format = 'png')
plt.close(fig)
plt.clf()
plt.cla()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:57:14 2021

@author: Haad-Rathore
"""

from scipy.io import loadmat, savemat      # Import function to read data
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import colors as cl
import sys, os, shutil
import numpy as np
import pandas as pd

# Load experimental summary data
dir_expsummary = '/run/media/hyr2/Data/Data/FH-BC2/6-18-2021/ePhys/awake/data/Processed/Saved_Data/exp_summary.xlsx'
df_exp_summary = pd.read_excel(dir_expsummary)
arr_exp_summary = df_exp_summary.to_numpy()

stim_start_time = arr_exp_summary[2,2]  # Stimulation start
Ntrials = arr_exp_summary[2,4]          # Number of trials
stim_end_time = arr_exp_summary[2,1] + stim_start_time  # End time of stimulation
time_seq = arr_exp_summary[2,0]         # Time of one sequence in seconds
Seq_perTrial =  arr_exp_summary[2,3]    # Number of sequences per trial

# Set parameters here
t_start = stim_start_time - 1
t_end = stim_end_time + 3.8
baseline_ROI_index = np.array([4,5,7])                      # indices of ROI that are to be selected as baseline
stim_start_time = stim_start_time - 0.6
scaling = 1                                                # The scaling factor b/w electrical and optical
save_filename_png = '/run/media/hyr2/Data/Data/FH-BC2/6-18-2021/ePhys_CBF/Chan_14.png'
save_filename_eps = '/run/media/hyr2/Data/Data/FH-BC2/6-18-2021/ePhys_CBF/Chan_14.eps'
save_filename_svg = '/run/media/hyr2/Data/Data/FH-BC2/6-18-2021/ePhys_CBF/Chan_14.svg'


# Loading Optical data
dict_optical = loadmat('/run/media/hyr2/Data/Data/FH-BC2/6-18-2021/CBF/Processed_Full-Analysis/data.mat')
rICT = dict_optical['rICT']
t_optical = dict_optical['t']
t_optical = np.reshape(t_optical,(t_optical.shape[1],))
baseline_ROI_index = baseline_ROI_index - 1
rICT_bsl = rICT[:,baseline_ROI_index]
rICT_bsl = np.mean(rICT_bsl,axis = 1)
rICT_bsl = np.reshape(rICT_bsl,(rICT_bsl.size,1))
rICT = np.subtract(rICT, rICT_bsl)

# Loading electrical data
dict_electrical = loadmat('/run/media/hyr2/Data/Data/FH-BC2/6-18-2021/ePhys/awake/data/Processed/Saved_Data/Chan14.mat')
LFP_ndPSD = dict_electrical['LFP_ndPSD']
LFP_ndPSD = LFP_ndPSD[0,:,:]
MUA_ndPSD = dict_electrical['MUA_ndPSD']
f_LFP = dict_electrical['f_LFP']
t_electrical = dict_electrical['time']
t_electrical = np.reshape(t_electrical,(t_electrical.shape[1],))
t_electrical = t_electrical - 0.45
MUA_ndPSD = np.reshape(MUA_ndPSD,(MUA_ndPSD.shape[1],))
f_LFP = np.reshape(f_LFP,(f_LFP.shape[1],))


# Getting the time axis
t_axis_electrical = np.where(np.logical_and(t_electrical>=t_start, t_electrical<=t_end))                     
t_axis_electrical = np.asarray(t_axis_electrical)
t_axis_electrical = np.reshape(t_axis_electrical,(t_axis_electrical.size,))
t_axis_optical = np.where(np.logical_and(t_optical>=t_start, t_optical<=t_end))                     
t_axis_optical = np.asarray(t_axis_optical)
t_axis_optical = np.reshape(t_axis_optical,(t_axis_optical.size,))

t_axis_electrical_final = np.linspace(int(t_electrical[t_axis_electrical[0]]),int(t_electrical[t_axis_electrical[-1]]), num = t_axis_electrical.size)
t_axis_optical_final = np.linspace(int(t_optical[t_axis_optical[0]]),int(t_optical[t_axis_optical[-1]]), num = t_axis_optical.size)

# Aligning the ePhys and CBF data
rICT = scaling * rICT[t_axis_optical,:]
rICT = np.transpose(rICT)
MUA_ndPSD = MUA_ndPSD[int(t_axis_electrical[0]):int(t_axis_electrical[-1]+1)]
LFP_ndPSD = LFP_ndPSD[:,int(t_axis_electrical[0]):int(t_axis_electrical[-1]+1)]
# rICT = rICT[[t_axis_optical],1]

# Aesthetic time scale
t_axis_optical_final = t_axis_optical_final - stim_start_time
t_axis_electrical_final = t_axis_electrical_final - stim_start_time
t = np.linspace(0,199e-3,100);
ADC = 0.5 * signal.square(2 * np.pi * 10 * (t+500e-3)) + 0.5
z = np.zeros((210,))
A1 = np.concatenate((z,ADC,z,z,z,z,z,z,z,z,z))
t = np.linspace(-0.48,5,210*10+100)

fg, (a3,a1, ax2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1,3, 2]})
a3.plot(t,A1, linewidth = 1.75)           # ADC signal
a3.set_xlim([t_axis_optical_final[0],t_axis_optical_final[-1]])
a3.set_ylabel('Trigger',fontsize = 14)
# a2.plot(t_axis_optical_final,rICT[0,:],color = 'r')
a2 = ax2.twinx()
a2.plot(t_axis_optical_final,rICT[1,:],color = 'g',linewidth = 1.8)
a2.set_ylim([-0.01,0.025])
ax2.vlines(0,-0.1,np.amax(MUA_ndPSD),linestyles = 'dashed', lw = 1.7 ,colors = 'k')
color = 'r'
ax2.set_ylabel('MUA  '+ r'$\Delta$' + r'$P_n$', color = color, fontsize=14)
ax2.plot(t_axis_electrical_final, MUA_ndPSD, color=color, linewidth = 1.8)
ax2.tick_params(axis='y', labelcolor=color)
a1.set_ylabel('Freq (Hz)', fontsize=14)
ax2.set_xlabel('Time (s)', fontsize=14)
a2.set_ylabel('$rICT_n$', fontsize=14)
a2.legend(['R1'], loc = 'upper right')

# ax2.legend(['MUA'], loc = 9)
plt.setp(ax2, xlim=(t_axis_electrical_final[0],t_axis_electrical_final[-1]))
im = a1.imshow(LFP_ndPSD,interpolation = 'hanning',aspect = 'auto',vmin = -0.5, vmax = 2.8, extent = [t_axis_electrical_final[0],t_axis_electrical_final[-1],f_LFP[1],f_LFP[-2]], origin = 'lower')
a1.vlines(0,20,150, linestyles = 'dashed', lw = 1.7, colors = 'k')
cbaxes = fg.add_axes([0.85, 0.425, 0.02, 0.35]) 
cb = fg.colorbar(im,ax = [a2],cax = cbaxes,label = r'$\Delta$'+r'$P_n$')
fg.set_size_inches((6, 6), forward = True)
fg.tight_layout() 

plt.savefig(save_filename_png,format = 'png',dpi = 600)
# plt.savefig(save_filename_eps,format = 'eps', dpi = 600)
plt.savefig(save_filename_svg,format = 'svg',dpi = 600)






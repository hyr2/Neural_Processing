# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:07:16 2021

@author: Haad-Rathore
"""

from scipy.io import loadmat       # Import function to read data.
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from scipy import signal
import sys, os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from natsort import natsorted
from Support import *

# Files and folders
source_dir = input('Enter the directory containing the output of Bin2Trials.py: \n')
output_dir = os.path.join(source_dir,'ERP-LFP')
if os.path.isdir(output_dir):
	os.rmdir(output_dir)        # delete directory if it already exists
source_dir_list = natsorted(os.listdir(source_dir))
os.mkdir(output_dir)
exp_summary_dir =  os.path.join(source_dir, source_dir_list[source_dir_list.index('exp_summary.xlsx')])
del source_dir_list[-1]
# Extracting data from summary file .xlsx
df_exp_summary = pd.read_excel(exp_summary_dir)
arr_exp_summary = df_exp_summary.to_numpy()
Num_chan = arr_exp_summary[0,0]         # Number of channels
Notch_freq = arr_exp_summary[0,1]       # Notch frequencey selected (in Hz)
Fs = arr_exp_summary[0,2]               # Sampling freq (in Hz)
stim_start_time = arr_exp_summary[2,2]  # Stimulation start
Ntrials = arr_exp_summary[2,4]          # Number of trials
stim_end_time = arr_exp_summary[2,1] + stim_start_time  # End time of stimulation
chan_map = arr_exp_summary[4:,0]

time_limit = int((stim_end_time+1)*Fs)		# Truncating analysis time to save computation time

# Extracting data from .csv files
for iter_chan in range(Num_chan):
	filename = os.path.join(source_dir,source_dir_list[iter_chan])
	df_chan = pd.read_csv(filename,dtype = np.single)
	arr_chan = df_chan.to_numpy()
	# Reading data from .csv
	ADC_data = arr_chan[:time_limit,-2]
	Time = arr_chan[:time_limit,-1]
	eEEG = arr_chan[:time_limit,:Ntrials]
	eEEG = np.transpose(eEEG)

	# Filtering
	eEEG_filtered_LFP = filterSignal_Notch(filterSignal_LFP_FIR(eEEG,Fs),Fs)		# Notch and LFP
	eEEG_filtered_Gamma = filterSignal_Notch(filterSignal_LFP_Gamma_FIR(eEEG,Fs),Fs) # Notch and Gamma band

	# Downsample
# 	eEEG_LFP = signal.decimate(eEEG,4, axis = 1)
	# eEEG_Gamma = signal.decimate(eEEG,4, axis = 1)
	# Computing colorbar range
	stat_mean = np.mean(eEEG_filtered_LFP)
	stat_std = np.std(eEEG_filtered_LFP)

	stat_mean_gamma = np.mean(eEEG_filtered_Gamma)
	stat_std_gamma = np.std(eEEG_filtered_Gamma)


	# Saving figure
	f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})
	a0.set_ylabel('Trial #')
	a1.set_ylabel('Trigger (V)')
	a1.set_xlabel('Time (s)')
	title_str = 'LFP Event-Related-Potential Channel #' + str(chan_map[iter_chan])
	f.suptitle(title_str)
	im = a0.imshow(eEEG_filtered_LFP,cmap='GnBu',extent=[Time[0],Time[-1], 1 , Ntrials],aspect = 'auto',origin = 'lower', vmax = stat_mean + 2*stat_std, vmin = stat_mean - 1.5*stat_std)
	a0.set_xlim([0,time_limit/Fs])
	a1.plot(Time,ADC_data)
	a1.set_xlim([0,time_limit/Fs])
	a0.vlines(stim_start_time,1,Ntrials,'r',linestyles = 'dashed', lw=1.5)
	a0.vlines(stim_end_time,1,Ntrials,'r',linestyles = 'dashed', lw=1.5)
	f.colorbar(im, ax = [a0,a1])
	filename_save = 'ERP-LFP-Channel' + str(chan_map[iter_chan]) + '.pdf'
	filename_save = os.path.join(output_dir,filename_save)
	f.set_size_inches((9, 6), forward=False)
	plt.savefig(filename_save,format = 'pdf')
	plt.close(f)
	# Saving figure
	f1, (a2, a3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})
	a2.set_ylabel('Trial #')
	a3.set_ylabel('Trigger (V)')
	a3.set_xlabel('Time (s)')
	title_str = 'Gamma band Event-Related-Potential Channel #' + str(chan_map[iter_chan])
	f1.suptitle(title_str)
	im = a2.imshow(eEEG_filtered_Gamma,cmap='GnBu',extent=[Time[0],Time[-1], 1 , Ntrials],aspect = 'auto',origin = 'lower', vmax = stat_mean_gamma + 2*stat_std_gamma, vmin = stat_mean_gamma - 1.5*stat_std_gamma)
	a2.set_xlim([0,time_limit/Fs])
	a3.plot(Time,ADC_data)
	a3.set_xlim([0,time_limit/Fs])
	a2.vlines(stim_start_time,1,Ntrials,'r',linestyles = 'dashed', lw=1.5)
	a2.vlines(stim_end_time,1,Ntrials,'r',linestyles = 'dashed', lw=1.5)
	f1.colorbar(im, ax = [a2,a3])
	filename_save = 'ERP-Gamma-Channel' + str(chan_map[iter_chan]) + '.pdf'
	filename_save = os.path.join(output_dir,filename_save)
	f1.set_size_inches((9, 6), forward=False)
	plt.savefig(filename_save,format = 'pdf')
	plt.close(f1)

	plt.clf()
	# Print progress
	print(int((iter_chan+1)/len(source_dir_list) * 100),'% done')
# 	plt.show()


# TO DO:
# 1. Save Gamma band data
# 2. Save lower band data

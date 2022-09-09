# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:07:16 2021

@author: Haad-Rathore
"""

# Notes: 
# Baseline time is selected to be 700 ms. It should not be too long since electrical activity can change very fast and very dramatically.



from scipy.io import loadmat, savemat      # Import function to read data.
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from scipy import signal, integrate, stats
import sys, os, shutil, pywt
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from natsort import natsorted
from Support import *

def Spectrogram_main(source_dir, pk_thresh, normalize_PSD_flag, t_activation_window, chan_map_knob):
	
	# INPUT ARGUMENTS:
	#   source_dir: The directory location of source file. Please run this code after running the BinChan.py and Bin2Trials.py
	#   pk_thresh : Threshold for peak detection from PSD calculations (USE 25 for Anesthetized and 2.5 for Awake)
	#   normalize_PSD_flag: Do you want to normalize the PSD calculations with the baseline PSD
	
	# Files and folders
	# source_dir = input('Enter the source directory: \n')
	output_dir_1 = os.path.join(source_dir,'Processed','Spectrogram')
	output_dir_2 = os.path.join(source_dir,'Processed','Rasters')
	output_dir_3 = os.path.join(source_dir,'Processed','Avg_Trials')
	filename_save_data = os.path.join(source_dir,'Processed','Spectrogram_mat')
	output_dir_cortical_depth = os.path.join(source_dir,'Processed','Cortical-Depth')
	dir_chan_list = os.path.join(source_dir,'chan_list.xlsx')
	if (chan_map_knob == 1):
		dir_chan_map = os.path.join(source_dir,'chan_map_1x32_128ch.xlsx')
	elif (chan_map_knob == 2):
		dir_chan_map = os.path.join(source_dir,'chan_map_2x16_128ch.xlsx')
	dir_Bin2Trials = os.path.join(source_dir,'Bin2Trials')
	dir_expsummary = os.path.join(source_dir,'exp_summary.xlsx')
	
	os.makedirs(output_dir_1, exist_ok=True)
	os.makedirs(output_dir_2, exist_ok=True)
	os.makedirs(output_dir_3, exist_ok=True)
	os.makedirs(filename_save_data, exist_ok=True)
	os.makedirs(output_dir_cortical_depth, exist_ok=True)
	
	# Extracting data from summary file .xlsx
	df_exp_summary = pd.read_excel(dir_expsummary)
	arr_exp_summary = df_exp_summary.to_numpy()
	Num_chan = arr_exp_summary[0,0]         # Number of channels
	Notch_freq = arr_exp_summary[0,1]       # Notch frequencey selected (in Hz)
	Fs = arr_exp_summary[0,2]               # Sampling freq (in Hz)
	stim_start_time = arr_exp_summary[4,1]-0.05   # Stimulation start - 50ms of window
	stim_start_time_original = arr_exp_summary[2,2]# original stimulation start time
	n_stim_start = int(Fs * stim_start_time)# Stimulation start time in samples
	Ntrials = arr_exp_summary[2,4]          # Number of trials
	stim_end_time = arr_exp_summary[2,1] + stim_start_time  # End time of stimulation
	time_seq = arr_exp_summary[4,0]         # Time of one sequence in seconds
	Seq_perTrial =  arr_exp_summary[2,3]    # Number of sequences per trial
	total_time = time_seq * Seq_perTrial    # Total time of the trial
	print('Each sequence is: ', time_seq, 'sec')
	time_seq = int(np.ceil(time_seq * Fs/2) * 2)                # Time of one sequence in samples (rounded up to even)
	
	# Extracting channel mapping info
	if chan_map_knob == 1:
		df_chanMap = pd.read_excel(dir_chan_map,usecols=[0,1,2,3],header = None,sheet_name = 2)
	elif (chan_map_knob == 2):
		df_chanMap = pd.read_excel(dir_chan_map,usecols=[0,1,2,3,4,5,6,7],header = None,sheet_name = 2)
	arr_chanMap = df_chanMap.to_numpy()                 # 4 shank device 1x32 channels on each shank
	df_chanList = pd.read_excel(dir_chan_list,header = 0)
	chan_list = df_chanList.to_numpy()
	chan_list = chan_list[:,0]
	chan_list = np.reshape(chan_list,(Num_chan,))
	
	# First channel only 
	list_Bin2Trials = natsorted(os.listdir(dir_Bin2Trials))
	filename = os.path.join(dir_Bin2Trials,list_Bin2Trials[0])
	df_chan = pd.read_csv(filename,dtype = np.single)
	arr_chan = df_chan.to_numpy()
	Time = arr_chan[:,-1]
	ADC_data = arr_chan[:,-2]
	total_time = len(Time)               # in samples
	
	# --------------------- SET THESE PARAMETERS ------------------------------                  
	time_window = 20e-3                     # Selecting a 50 ms time window for STFT
	n_time_window = int(time_window * Fs)   # Time in samples
	n_chan = 128                            # Total channels in the device
	n_density = 32                          # Number of channels on a single shank
	electrode_spacing = 25                  # Spacing in microns b/w electrodes
	skip_trials = np.array([0,1,2,3,4,5],dtype = np.int16) # Skip these trials
	time_limit_spectrogram = stim_end_time + 1          # Specify the time limit (in seconds) for the plot of the spectrogram
	time_start_spectrogram = stim_start_time - 1    # For visual plotting (because bin2trial.py had 0.5 s extra)
	decimate_factor = 20                            # Factor for decimating for Morlet Transform
	# f = pywt.scale2frequency('cmor2-1.5',np.arange(15,250,1)) * Fs/decimate_factor     # Please check this to get the correct freuqncy band      
	Ntrials_copy = Ntrials - skip_trials.size
	
	
	# Cortical depth plots
	MUA_depth_peak = np.zeros((arr_chanMap.shape[0],arr_chanMap.shape[1],2))
	MUA_depth_peak[:] = np.nan
	MUA_depth_mean = np.zeros(arr_chanMap.shape)
	MUA_depth_mean[:] = np.nan
	LFP_depth_peak = np.zeros((arr_chanMap.shape[0],arr_chanMap.shape[1],4))
	LFP_depth_peak[:] = np.nan
	LFP_depth_mean = np.zeros((arr_chanMap.shape[0],arr_chanMap.shape[1],4))
	LFP_depth_mean[:] = np.nan
	MUA_depth_mean_post = np.zeros(arr_chanMap.shape)
	MUA_depth_mean_post[:] = np.nan
	LFP_depth_mean_post = np.zeros((arr_chanMap.shape[0],arr_chanMap.shape[1],4))
	LFP_depth_mean_post[:] = np.nan
	
	
	# Iterate over channels -------------------------------------------------
	iter_progress = 0
	for iter_chan in chan_list:
		iter_chan = np.int(iter_chan)
		# finding location of this channel from the channel map
		# first element is depth index and second element is shank index
		chan_loc = np.reshape(np.where(iter_chan == arr_chanMap),(2,))
		filename_str = 'Chan' + str(iter_chan) + '.csv'
		filename = os.path.join(dir_Bin2Trials,filename_str)
		df_chan = pd.read_csv(filename,dtype = np.single)
		arr_chan = df_chan.to_numpy()
		eEEG = arr_chan[:,:Ntrials]             # Electrical recording
		eEEG = np.transpose(eEEG)
		eEEG = eEEG[:,0:int(time_limit_spectrogram*Fs)]
		# Filtering signals
		eEEG_filtered = filterSignal_notch(eEEG,Fs,60, axis_value = 1)                  # 60 Hz Notch  
		eEEG_filtered = filterSignal_notch(eEEG_filtered,Fs,120, axis_value = 1)        # 120 Hz Notch
		eEEG_filtered = filterSignal_notch(eEEG_filtered,Fs,180, axis_value = 1)        # 120 Hz Notch
		eEEG_MUA = filterSignal_MUA(eEEG,Fs,axis_value = 1)                             # MUA 0.3-3 Khz
		eEEG_filtered = filterSignal_lowpassLFP(eEEG_filtered,Fs, axis_value = 1)       # LFP (13-160 Hz)
		
		# Compute power spectral density from short-time FT (MUA ONLY)
		# f, t, Sxx = compute_PSD(eEEG_MUA, Fs, n_time_window, axis_value = 1)
# 		f_MUA, t_stft, Sxx_MUA = compute_PSD(eEEG_MUA, Fs, n_time_window, axis_value = 1)
# 		f_MUA = np.where(np.logical_and(f_MUA>=350, f_MUA<3000))
# 		f_MUA = np.asarray(f_MUA)
# 		f_MUA = np.reshape(f_MUA,(f_MUA.size,))
# 		Time_mwt_MUA = t_stft
# 		cwt_arr_MUA = Sxx_MUA

		if (Fs == 20e3):
			eEEG_MUA = signal.decimate(eEEG_MUA, 2, ftype = 'iir', axis = 1)
			cwt_arr_MUA, f_MUA = pywt.cwt(eEEG_MUA,np.arange(10,100,10),'cmor0.005-3.0',sampling_period = 2/Fs, method = 'fft', axis = 1)
		elif (Fs == 25e3):
			eEEG_MUA = signal.decimate(eEEG_MUA, 2, ftype = 'iir', axis = 1)
			cwt_arr_MUA, f_MUA = pywt.cwt(eEEG_MUA,np.arange(12,115,10),'cmor0.005-3.0',sampling_period = 2/Fs, method = 'fft', axis = 1)         
		
		cwt_arr_MUA = np.transpose(cwt_arr_MUA,axes = [1,0,2])
		cwt_arr_MUA = np.abs(cwt_arr_MUA)
		cwt_arr_MUA = np.mean(cwt_arr_MUA, axis = 1)
	#   cwt_arr_MUA = np.abs(eEEG_MUA)             # EXTRA FOR LAN
		Time_mwt_MUA = np.linspace(0,time_limit_spectrogram,eEEG_MUA.shape[1])
		t_baseline = np.where(np.logical_and(Time_mwt_MUA>=(stim_start_time-1), Time_mwt_MUA<=stim_start_time-0.2))
		t_baseline = np.asarray(t_baseline)
		t_baseline = np.reshape(t_baseline,(t_baseline.size,))
		# Compute power spectral density Morlet Wavelet Transform (LFP ONLY)
		# frequencies = pywt.scale2frequency('cmor0.005-3.0',np.arange(10,100,10)) / dt
		eEEG_filtered = signal.decimate(eEEG_filtered, decimate_factor, ftype = 'iir', axis = 1)          # decimation to reduce complexity
	#   cwt_arr, freq = pywt.cwt(eEEG_filtered,np.arange(3.50,60,0.1),'cmor2.0-0.5',sampling_period = decimate_factor/Fs, method = 'conv', axis = 1)
	
		if (Fs == 20e3):
			cwt_arr, freq = pywt.cwt(eEEG_filtered,np.arange(20,400,1),'cmor0.1-3.0',sampling_period = decimate_factor/Fs, method = 'fft', axis = 1)
		elif (Fs == 25e3):
			cwt_arr, freq = pywt.cwt(eEEG_filtered,np.arange(25,480,1.25),'cmor0.1-3.0',sampling_period = decimate_factor/Fs, method = 'fft', axis = 1)
			
		cwt_arr = np.transpose(cwt_arr,axes = [1,0,2])
		cwt_arr = np.abs(cwt_arr)
		Time_mwt = np.linspace(0,time_limit_spectrogram,eEEG_filtered.shape[1])
		
		t_baseline_cwt = np.where(np.logical_and(Time_mwt>=(stim_start_time-1), Time_mwt<=stim_start_time-0.2))
		t_baseline_cwt = np.asarray(t_baseline_cwt)
		t_baseline_cwt = np.reshape(t_baseline_cwt,(t_baseline_cwt.size,))
		
	#   freq_axis = np.where(np.logical_and(freq>=8, freq<=140)) 
	#   freq_axis = np.asarray(freq_axis)
	#   freq_axis = np.reshape(freq_axis,(freq_axis.size,))
		# coeff = stats.zscore(coeff,axis = 2)
		# coeff = np.mean(coeff, axis = 1)
		
		#------------- LFP Alpha Band
		f_Alpha = np.where((freq>=12) & (freq<15))               # finding the indices of 8-15 Hz band
		f_Alpha = np.asarray(f_Alpha)
		f_Alpha = np.reshape(f_Alpha,(f_Alpha.size,))
		arr_ndPSD_Alpha = np.zeros((Ntrials,Time_mwt.size),dtype=np.single)
		#------------- LFP Beta Band
		f_Beta = np.where((freq>=16) & (freq<32))               # finding the indices of 16-32 Hz band
		f_Beta = np.asarray(f_Beta)
		f_Beta = np.reshape(f_Beta,(f_Beta.size,))
		arr_ndPSD_Beta = np.zeros((Ntrials,Time_mwt.size),dtype=np.single)
		#------------- LFP Gamma band
		f_Gamma = np.where((freq>=32) & (freq<100))             # Find the indices of the Gamma band (40-100 Hz)
		f_Gamma = np.asarray(f_Gamma)
		f_Gamma = np.reshape(f_Gamma,(f_Gamma.size,))
		arr_ndPSD_Gamma = np.zeros((Ntrials,Time_mwt.size),dtype=np.single)
		#------------- LFP High frequency
		f_high = np.where((freq>=100) & (freq<=140))               # finding the indices of 100-140 Hz LFP band
		f_high = np.asarray(f_high)
		f_high = np.reshape(f_high,(f_high.size,))
		arr_ndPSD_high = np.zeros((Ntrials,Time_mwt.size),dtype=np.single)
	
		arr_ndPSD = np.zeros((Ntrials,freq.size,Time_mwt.size),dtype=np.single)        # for LFP
		arr_ndPSD_MUA = np.zeros((Ntrials,Time_mwt_MUA.size),dtype=np.single)      # for MUA
		arr_FFT_MUA_6Hz_noise = np.zeros([Ntrials,], dtype = np.single)           # used for noise analysis
		new_skip_trials = skip_trials
		# Iterating over trials
		if (skip_trials.size == 0):
			for iter_trial in range(0,Ntrials):
	#           psd_bl = np.mean(Sxx[iter_trial,:,t_baseline[0:]],axis = 0)         # Taking average over time
				psd_bl_MUA = np.mean(cwt_arr_MUA[iter_trial,t_baseline[0:]],axis = 0)
	#           psd_bl = np.reshape(psd_bl,(psd_bl.size,1))
	#           psd_bl_MUA = np.reshape(psd_bl_MUA,(psd_bl_MUA.size,1))
				cwt_bsl = np.mean(cwt_arr[iter_trial,:,t_baseline_cwt[0:]], axis = 0)
				cwt_bsl = np.reshape(cwt_bsl,(cwt_bsl.size,1))
				if normalize_PSD_flag == 1:
					# compute normalized change in PSD
		#           ndPSD = (Sxx[iter_trial,:,:] - psd_bl)/psd_bl                       # Normalzied change in PSD/Hz
					ndPSD_MUA = (cwt_arr_MUA[iter_trial,:] - psd_bl_MUA)/psd_bl_MUA     # Normalzied change in PSD/Hz for MUA
					# compute normalized change in  Morlet Transform
					nd_cwt = (cwt_arr[iter_trial,:,:] - cwt_bsl)/cwt_bsl                # Normalized change in cw morlet transform
				else:
					# compute change in PSD
		#           ndPSD = (Sxx[iter_trial,:,:] - psd_bl)/psd_bl                       # change in PSD/Hz
					ndPSD_MUA = (cwt_arr_MUA[iter_trial,:] - psd_bl_MUA)                # change in PSD/Hz for MUA
					# compute normalized change in  Morlet Transform
					nd_cwt = (cwt_arr[iter_trial,:,:] - cwt_bsl)                        # change in cw morlet transform
				# Average values of the corresponding frequency bands being analyzed
				ndPSD_Alpha = np.mean(nd_cwt[f_Alpha[0]:f_Alpha[-1]+1,:],axis = 0)
				ndPSD_Beta = np.mean(nd_cwt[f_Beta[0]:f_Beta[-1]+1,:],axis = 0)
				ndPSD_Gamma = np.mean(nd_cwt[f_Gamma[0]:f_Gamma[-1]+1,:],axis = 0)
				ndPSD_high = np.mean(nd_cwt[f_high[0]:f_high[-1]+1,:],axis = 0)
				# storing in big array
				arr_ndPSD_Alpha[iter_trial,:] = ndPSD_Alpha
				arr_ndPSD_Beta[iter_trial,:] = ndPSD_Beta
				arr_ndPSD_Gamma[iter_trial,:] = ndPSD_Gamma
				arr_ndPSD_high[iter_trial,:] = ndPSD_high
				# for spectrogram_ndPSD and spectrogram_ndPSD_MUA and continuous wavelet transform
				arr_ndPSD_MUA[iter_trial,:] = ndPSD_MUA
				arr_ndPSD[iter_trial,:,:] = nd_cwt
		else:
			for iter_trial in range(0,Ntrials):
				if not (iter_trial == skip_trials).any():
					# psd_bl = np.mean(Sxx[iter_trial,:,t_baseline[0:]],axis = 0)         # Taking average over time
					psd_bl_MUA = np.mean(cwt_arr_MUA[iter_trial,t_baseline[0:]],axis = 0)
		#           psd_bl = np.reshape(psd_bl,(psd_bl.size,1))
		#           psd_bl_MUA = np.reshape(psd_bl_MUA,(psd_bl_MUA.size,1))
					cwt_bsl = np.mean(cwt_arr[iter_trial,:,t_baseline_cwt[0:]], axis = 0)
					cwt_bsl = np.reshape(cwt_bsl,(cwt_bsl.size,1))
					if normalize_PSD_flag == 1:
						# compute normalized change in PSD
			#           ndPSD = (Sxx[iter_trial,:,:] - psd_bl)/psd_bl                       # Normalzied change in PSD/Hz
						ndPSD_MUA = (cwt_arr_MUA[iter_trial,:] - psd_bl_MUA)/psd_bl_MUA       # Normalzied change in PSD/Hz for MUA
						# compute normalized change in  Morlet Transform
						nd_cwt = (cwt_arr[iter_trial,:,:] - cwt_bsl)/cwt_bsl                # Normalized change in cw morlet transform
					else:
						# compute change in PSD
			#           ndPSD = (Sxx[iter_trial,:,:] - psd_bl)/psd_bl                       # change in PSD/Hz
						ndPSD_MUA = (cwt_arr_MUA[iter_trial,:] - psd_bl_MUA)          # change in PSD/Hz for MUA
						# compute normalized change in  Morlet Transform
						nd_cwt = (cwt_arr[iter_trial,:,:] - cwt_bsl)                # change in cw morlet transform
					# Average values of the corresponding frequency bands being analyzed
					ndPSD_Alpha = np.mean(nd_cwt[f_Alpha[0]:f_Alpha[-1]+1,:],axis = 0)
					ndPSD_Beta = np.mean(nd_cwt[f_Beta[0]:f_Beta[-1]+1,:],axis = 0)
					ndPSD_Gamma = np.mean(nd_cwt[f_Gamma[0]:f_Gamma[-1]+1,:],axis = 0)
					ndPSD_high = np.mean(nd_cwt[f_high[0]:f_high[-1]+1,:],axis = 0)
					# Further filtering to remove low frequency noise (maybe heart beat or breathing)
# 					ndPSD_Alpha = filterSignal_High(ndPSD_Alpha, Fs/20, 6.5)         # No need to filter Alpha band since its robust to the 6Hz noise
					ndPSD_Beta = filterSignal_High(ndPSD_Beta, Fs/20, 6.5)
					ndPSD_Gamma = filterSignal_High(ndPSD_Gamma, Fs/20, 6.5)
					ndPSD_high = filterSignal_High(ndPSD_high, Fs/20, 6.5)
					ndPSD_MUA = filterSignal_High(ndPSD_MUA, Fs/2, 6.5)
# 					nd_cwt = filterSignal_High(nd_cwt, Fs/20, 6.5, axis_value = 1)
					# storing in big array
					arr_ndPSD_Alpha[iter_trial,:] = ndPSD_Alpha
					arr_ndPSD_Beta[iter_trial,:] = ndPSD_Beta
					arr_ndPSD_Gamma[iter_trial,:] = ndPSD_Gamma
					arr_ndPSD_high[iter_trial,:] = ndPSD_high
					arr_ndPSD_MUA[iter_trial,:] = ndPSD_MUA
					arr_ndPSD[iter_trial,:,:] = nd_cwt

					# Optional plotting for debugging
# 					_, vec_FFT = ShowFFT(ndPSD_MUA,Fs/2, 0)
# 					arr_FFT_MUA_6Hz_noise[iter_trial] = np.mean(vec_FFT[29:55])
# 					if (np.mean(vec_FFT[29:55]) < 2.7):
# 						# for spectrogram_ndPSD and spectrogram_ndPSD_MUA and continuous wavelet transform
# 						arr_ndPSD_MUA[iter_trial,:] = ndPSD_MUA
# 						arr_ndPSD[iter_trial,:,:] = nd_cwt
# 					else:
# 						new_skip_trials = np.append(new_skip_trials,iter_trial)

			# delete empty trials
# 			plt.plot(arr_FFT_MUA_6Hz_noise)
			arr_ndPSD_Alpha = np.delete(arr_ndPSD_Alpha,new_skip_trials,0)
			arr_ndPSD_Beta = np.delete(arr_ndPSD_Beta,new_skip_trials,0)
			arr_ndPSD_Gamma = np.delete(arr_ndPSD_Gamma,new_skip_trials,0)
			arr_ndPSD_high = np.delete(arr_ndPSD_high,new_skip_trials,0)
			arr_ndPSD_MUA = np.delete(arr_ndPSD_MUA,new_skip_trials,0)
			arr_ndPSD = np.delete(arr_ndPSD,new_skip_trials,0)
	
		# Averaging across trials
		avg_ndPSD_Alpha = np.mean(arr_ndPSD_Alpha,axis=0)
		avg_ndPSD_Beta = np.mean(arr_ndPSD_Beta,axis=0)
		avg_ndPSD_Gamma = np.mean(arr_ndPSD_Gamma,axis=0)
		avg_ndPSD_high = np.mean(arr_ndPSD_high,axis=0)
	
		# Averaging across trials (true spectrogram)
		f_axis = np.where((freq>=12) & (freq<=140))
		f_axis = np.asarray(f_axis)
		f_axis = np.reshape(f_axis,(f_axis.size,))
		spectrogram_ndPSD = np.mean(arr_ndPSD[:,f_axis[0]:f_axis[-1]+1,:],axis = 0)
		f_axis_MUA = np.where((f_MUA>=300) & (f_MUA<=3000))
		f_axis_MUA = np.asarray(f_axis_MUA)
		f_axis_MUA = np.reshape(f_axis_MUA,(f_axis_MUA.size,))
		spectrogram_ndPSD_MUA = np.mean(arr_ndPSD_MUA, axis=0)
		# spectrogram_ndPSD = np.transpose(spectrogram_ndPSD)
	
		# Activation Time windows
		t_activation = np.where(np.logical_and(Time_mwt>=stim_start_time, Time_mwt<=stim_start_time + t_activation_window))
		t_activation = np.asarray(t_activation)
		t_activation = np.reshape(t_activation,(t_activation.size,))
		t_activation_MUA = np.where(np.logical_and(Time_mwt_MUA>=stim_start_time, Time_mwt_MUA<=stim_start_time + t_activation_window))
		t_activation_MUA = np.asarray(t_activation_MUA)
		t_activation_MUA = np.reshape(t_activation_MUA,(t_activation_MUA.size,))
		MUA_depth_mean[chan_loc[0],chan_loc[1]] = np.mean(spectrogram_ndPSD_MUA[t_activation_MUA[0:]])
		# MUA_depth_peak[chan_loc[0],chan_loc[1]] = np.amax(spectrogram_ndPSD_MUA[t_activation_MUA[0:]])  
		MUA_depth_peak[chan_loc[0],chan_loc[1],:] =  detect_peak_basic(spectrogram_ndPSD_MUA[t_activation_MUA[0:]],500,pk_thresh)     
		LFP_depth_mean[chan_loc[0],chan_loc[1],0] = np.mean(avg_ndPSD_Alpha[t_activation[0:]])
		LFP_depth_peak[chan_loc[0],chan_loc[1],0] = np.amax(avg_ndPSD_Alpha[t_activation[0:]])
		LFP_depth_mean[chan_loc[0],chan_loc[1],1] = np.mean(avg_ndPSD_Beta[t_activation[0:]])
		LFP_depth_peak[chan_loc[0],chan_loc[1],1] = np.amax(avg_ndPSD_Beta[t_activation[0:]])
		LFP_depth_mean[chan_loc[0],chan_loc[1],2] = np.mean(avg_ndPSD_Gamma[t_activation[0:]])
		LFP_depth_peak[chan_loc[0],chan_loc[1],2] = np.amax(avg_ndPSD_Gamma[t_activation[0:]])
		LFP_depth_mean[chan_loc[0],chan_loc[1],3] = np.mean(avg_ndPSD_high[t_activation[0:]])
		LFP_depth_peak[chan_loc[0],chan_loc[1],3] = np.amax(avg_ndPSD_high[t_activation[0:]])
		# Post activation Time windows
		t_activation_post = np.where(np.logical_and(Time_mwt>=stim_end_time-time_window, Time_mwt<=stim_end_time+time_window))
		t_activation_post = np.asarray(t_activation_post)
		t_activation_post = np.reshape(t_activation_post,(t_activation_post.size,))
		LFP_depth_mean_post[chan_loc[0],chan_loc[1],0] = np.mean(avg_ndPSD_Alpha[t_activation_post[0:]])
		LFP_depth_mean_post[chan_loc[0],chan_loc[1],1] = np.mean(avg_ndPSD_Beta[t_activation_post[0:]])
		LFP_depth_mean_post[chan_loc[0],chan_loc[1],2] = np.mean(avg_ndPSD_Gamma[t_activation_post[0:]])
		LFP_depth_mean_post[chan_loc[0],chan_loc[1],3] = np.mean(avg_ndPSD_high[t_activation_post[0:]])
		
		# ---------- PLOTTING --------------------------------
		# Saving LFP-Alpha
		fg, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})
		fg.set_visible(1)
		a0.set_ylabel('Trial #')
		a1.set_ylabel('Trigger (V)')
		a1.set_xlabel('Time (s)')
		title_str = 'Raster-Alpha (8-15 Hz) | Channel #' + str(iter_chan)
		fg.suptitle(title_str)
		max_lim = np.mean(arr_ndPSD_Alpha) + 3*np.std(arr_ndPSD_Alpha)
		min_lim = np.mean(arr_ndPSD_Alpha) - 1*np.std(arr_ndPSD_Alpha)
		im = a0.imshow(arr_ndPSD_Alpha,cmap='YlGn_r',extent=[Time_mwt[0],Time_mwt[-1], 1 , Ntrials_copy],aspect = 'auto',origin = 'lower', vmax = max_lim, vmin = min_lim, interpolation = 'hamming')
		a0.set_xlim([time_start_spectrogram,time_limit_spectrogram])
		a1.plot(Time,ADC_data)
		a1.set_xlim([time_start_spectrogram,time_limit_spectrogram])
		a0.vlines(stim_start_time,1,Ntrials_copy,'r',linestyles = 'dashed', lw=1.5)
		a0.vlines(stim_end_time,1,Ntrials_copy,'r',linestyles = 'dashed', lw=1.5)
		fg.colorbar(im, ax = [a0,a1],label =  r'$\Delta$'+r'$P_n$')
		filename_save = 'Raster-Alpha' + str(iter_chan) + '.png'
		filename_save = os.path.join(output_dir_2,filename_save)
		fg.set_size_inches((9, 6), forward=False)
		plt.savefig(filename_save,format = 'png')
		plt.close(fg)
		plt.clf()
		plt.cla()
		
		# Saving LFP-Beta
		fg, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})
		fg.set_visible(1)
		a0.set_ylabel('Trial #')
		a1.set_ylabel('Trigger (V)')
		a1.set_xlabel('Time (s)')
		title_str = 'Raster-Beta (16-31 Hz) | Channel #' + str(iter_chan)
		fg.suptitle(title_str)
		max_lim = np.mean(arr_ndPSD_Beta) + 3*np.std(arr_ndPSD_Beta)
		min_lim = np.mean(arr_ndPSD_Beta) - 1*np.std(arr_ndPSD_Beta)
		im = a0.imshow(arr_ndPSD_Beta,cmap='YlGn_r',extent=[Time_mwt[0],Time_mwt[-1], 1 , Ntrials_copy],aspect = 'auto',origin = 'lower', vmax = max_lim, vmin = min_lim, interpolation = 'hamming')
		a0.set_xlim([time_start_spectrogram,time_limit_spectrogram])
		a1.plot(Time,ADC_data)
		a1.set_xlim([time_start_spectrogram,time_limit_spectrogram])
		a0.vlines(stim_start_time,1,Ntrials_copy,'r',linestyles = 'dashed', lw=1.5)
		a0.vlines(stim_end_time,1,Ntrials_copy,'r',linestyles = 'dashed', lw=1.5)
		fg.colorbar(im, ax = [a0,a1],label =  r'$\Delta$'+r'$P_n$')
		filename_save = 'Raster-Beta' + str(iter_chan) + '.png'
		filename_save = os.path.join(output_dir_2,filename_save)
		fg.set_size_inches((9, 6), forward=False)
		plt.savefig(filename_save,format = 'png')
		plt.close(fg)
		plt.clf()
		plt.cla()
	
		# Saving LFP-Gamma
		f1, (a2, a3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})
		f1.set_visible(1)
		a2.set_ylabel('Trial #')
		a3.set_ylabel('Trigger (V)')
		a3.set_xlabel('Time (s)')
		title_str = 'Raster-Gamma (32-100 Hz) | Channel #' + str(iter_chan)
		f1.suptitle(title_str)
		max_lim = np.mean(arr_ndPSD_Gamma) + 3*np.std(arr_ndPSD_Gamma)
		min_lim = np.mean(arr_ndPSD_Gamma) - 1*np.std(arr_ndPSD_Gamma)
		im = a2.imshow(arr_ndPSD_Gamma,cmap='YlGn_r',extent=[Time_mwt[0],Time_mwt[-1], 1 , Ntrials_copy],aspect = 'auto',origin = 'lower', vmax = max_lim, vmin = min_lim, interpolation = 'hamming')
		a2.set_xlim([time_start_spectrogram,time_limit_spectrogram])
		a3.plot(Time,ADC_data)
		a3.set_xlim([time_start_spectrogram,time_limit_spectrogram])
		a2.vlines(stim_start_time,1,Ntrials_copy,'r',linestyles = 'dashed', lw=1.5)
		a2.vlines(stim_end_time,1,Ntrials_copy,'r',linestyles = 'dashed', lw=1.5)
		a3.vlines(stim_start_time,1,5,'r',linestyles = 'dashed', lw=1.5)
		a3.vlines(stim_end_time,1,5,'r',linestyles = 'dashed', lw=1.5)
		f1.colorbar(im, ax = [a2,a3],label =  r'$\Delta$'+r'$P_n$')
		filename_save = 'Raster-Gamma' + str(iter_chan) + '.png'
		filename_save = os.path.join(output_dir_2,filename_save)
		f1.set_size_inches((9, 6), forward=False)
		plt.savefig(filename_save,format = 'png')
		plt.close(f1)
		plt.clf()
		plt.cla()
	
		# Saving LFP-high freq
		f1, (a2, a3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})
		f1.set_visible(1)
		a2.set_ylabel('Trial #')
		a3.set_ylabel('Trigger (V)')
		a3.set_xlabel('Time (s)')
		title_str = 'Raster-LFP-high (100-140 Hz) | Channel #' + str(iter_chan)
		f1.suptitle(title_str)
		max_lim = np.mean(arr_ndPSD_high) + 3*np.std(arr_ndPSD_high)
		min_lim = np.mean(arr_ndPSD_high) - 1*np.std(arr_ndPSD_high)
		im = a2.imshow(arr_ndPSD_high,cmap='YlGn_r',extent=[Time_mwt[0],Time_mwt[-1], 1 , Ntrials_copy],aspect = 'auto',origin = 'lower', vmax = max_lim, vmin = min_lim, interpolation = 'hamming')
		a2.set_xlim([time_start_spectrogram,time_limit_spectrogram])
		a3.plot(Time,ADC_data)
		a3.set_xlim([time_start_spectrogram,time_limit_spectrogram])
		a2.vlines(stim_start_time,1,Ntrials_copy,'r',linestyles = 'dashed', lw=1.5)
		a2.vlines(stim_end_time,1,Ntrials_copy,'r',linestyles = 'dashed', lw=1.5)
		a3.vlines(stim_start_time,1,5,'r',linestyles = 'dashed', lw=1.5)
		a3.vlines(stim_end_time,1,5,'r',linestyles = 'dashed', lw=1.5)
		f1.colorbar(im, ax = [a2,a3],label =  r'$\Delta$'+r'$P_n$')
		filename_save = 'Raster-LFP-high' + str(iter_chan) + '.png'
		filename_save = os.path.join(output_dir_2,filename_save)
		f1.set_size_inches((9, 6), forward=False)
		plt.savefig(filename_save,format = 'png')
		plt.close(f1)
		plt.clf()
		plt.cla()
		
		# Saving cwt morlet 
		f_cwt = freq[f_axis[:]]
		LL = len(f_cwt)
		LL_x = len(Time_mwt)
		x_tick_list = np.array(np.arange(0,time_limit_spectrogram*Fs/decimate_factor,1*Fs/decimate_factor),dtype = np.int16)
		x_label_list = np.array(np.arange(Time_mwt[0],time_limit_spectrogram,1),dtype = np.int16)
		y_tick_list = np.array([LL,17/20*LL,3/4*LL,LL/2,LL/4,LL/10,0],dtype = np.int16)
		y_label_list = np.around([f_cwt[-1],f_cwt[int(17*LL/20)],f_cwt[int(3/4*LL)],f_cwt[int(LL/2)],f_cwt[int(LL/4)],f_cwt[int(LL/10)],f_cwt[0]])
		fg, a0 = plt.subplots(1, 1)
		fg.set_visible(1)
		a0.set_ylabel('Freq (Hz)')
		a0.set_xlabel('Time (s)')
		title_str = 'Average across trials-Morlet-Wavelets' + ' | Channel #' + str(iter_chan)
		fg.suptitle(title_str)
		max_lim = np.mean(spectrogram_ndPSD) + 4*np.std(spectrogram_ndPSD)
		min_lim = np.mean(spectrogram_ndPSD) - 0.5*np.std(spectrogram_ndPSD)
		im = a0.imshow(spectrogram_ndPSD,cmap='jet', aspect = 'auto', origin = 'upper', vmax = max_lim, vmin = min_lim, interpolation = 'hamming')
		a0.set_yticks(y_tick_list)
		a0.set_yticklabels(y_label_list)
		a0.set_xticks(x_tick_list)
		a0.set_xticklabels(x_label_list)
		fg.colorbar(im, ax = a0,label =  r'$\Delta$'+r'$P_n$')
		filename_save = 'cwt-' + str(iter_chan) + '.svg'
		filename_save = os.path.join(output_dir_1,filename_save)
		fg.set_size_inches((9, 6), forward=False)
		plt.savefig(filename_save,format = 'svg')
		plt.close(fg)
		plt.clf()
		plt.cla()   
		
	
		# Saving Averaging across trials for the three bands
		fg, (a7,a8, a9, a10, a11) = plt.subplots(5, 1, gridspec_kw={'height_ratios': [1,1,1,1,1]})
		fg.set_visible(1)
		a7.set_ylabel(r'$\Delta$'+r'$P_n$')
		a8.set_ylabel(r'$\Delta$'+r'$P_n$')
		a9.set_ylabel(r'$\Delta$'+r'$P_n$')
		a10.set_ylabel(r'$\Delta$'+r'$P_n$')
		a11.set_ylabel(r'$\Delta$'+r'$P_n$')
		a11.set_xlabel('Time (s)')
		title_str = 'Average across trials-' +  r'$\Delta$'+r'$P_n$' ' | Channel #' + str(iter_chan)
		fg.suptitle(title_str)
		a7.plot(Time_mwt,avg_ndPSD_Alpha,'m', lw = 1.75)
		a8.plot(Time_mwt,avg_ndPSD_Beta,'y', lw = 1.75)
		a9.plot(Time_mwt,avg_ndPSD_Gamma,'b',lw = 1.75)
		a10.plot(Time_mwt,avg_ndPSD_high,'g',lw = 1.75)
		a11.plot(Time_mwt_MUA,spectrogram_ndPSD_MUA,'k', lw = 1.75)
		a7.legend([r'$\alpha$'])
		a8.legend([r'$\beta$'])
		a9.legend([r'$\gamma$'])
		a10.legend(['LFP-high'])
		a11.legend(['MUA'])
		a7.set_xlim([time_start_spectrogram+0.25,time_limit_spectrogram-0.25])
		a8.set_xlim([time_start_spectrogram+0.25,time_limit_spectrogram-0.25])
		a9.set_xlim([time_start_spectrogram+0.25,time_limit_spectrogram-0.25])
		a10.set_xlim([time_start_spectrogram+0.25,time_limit_spectrogram-0.25])
		a11.set_xlim([time_start_spectrogram+0.25,time_limit_spectrogram-0.25])
		max_lim = np.amax(avg_ndPSD_Alpha)
		a7.vlines(stim_start_time,-0.5,max_lim,'r',linestyles = 'dashed', lw=1.5)
		a7.vlines(stim_end_time,-0.5,max_lim,'r',linestyles = 'dashed', lw=1.5)
		max_lim = np.amax(avg_ndPSD_Beta)
		a8.vlines(stim_start_time,-0.5,max_lim,'r',linestyles = 'dashed', lw=1.5)
		a8.vlines(stim_end_time,-0.5,max_lim,'r',linestyles = 'dashed', lw=1.5)
		max_lim = np.amax(avg_ndPSD_Gamma)
		a9.vlines(stim_start_time,-0.5,max_lim,'r',linestyles = 'dashed', lw=1.5)
		a9.vlines(stim_end_time,-0.5,max_lim,'r',linestyles = 'dashed', lw=1.5)
		max_lim = np.amax(avg_ndPSD_high)
		a10.vlines(stim_start_time,-0.5,max_lim,'r',linestyles = 'dashed', lw=1.5)
		a10.vlines(stim_end_time,-0.5,max_lim,'r',linestyles = 'dashed', lw=1.5)
		max_lim = np.amax(spectrogram_ndPSD_MUA)
		min_lim = np.amin(spectrogram_ndPSD_MUA)
		a11.vlines(stim_start_time,min_lim,max_lim,'r',linestyles = 'dashed', lw=1.5)
		a11.vlines(stim_end_time,min_lim,max_lim,'r',linestyles = 'dashed', lw=1.5)
		filename_save = 'Average-change-PSD-LFP' + str(iter_chan) + '.svg'
		filename_save = os.path.join(output_dir_3,filename_save)
		fg.set_size_inches((9, 8), forward=False)
		plt.savefig(filename_save,format = 'svg')
		plt.close('all')
		plt.clf()
		plt.cla()
		
		# Saving to excel file and .mat files
		filename_save = 'Chan' + str(iter_chan) + '.mat'
		filename_save = os.path.join(filename_save_data,filename_save)
		data_summary = {'empty':[0],'time_MUA':Time_mwt_MUA,'time_LFP':Time_mwt,'freq_LFP':f_cwt,'LFP_ndPSD':spectrogram_ndPSD,'MUA_ndPSD':spectrogram_ndPSD_MUA,'Alpha_ndPSD':avg_ndPSD_Alpha,'Beta_ndPSD':avg_ndPSD_Beta,'Gamma_ndPSD':avg_ndPSD_Gamma,'LFPhigh_ndPSD':avg_ndPSD_high,'decimate_factor':decimate_factor}
		savemat(filename_save,data_summary)
		"{0:2d} % done".format(int((iter_chan+1)/len(chan_list) * 100))
		iter_progress = iter_progress + 1
		print((iter_progress)/len(chan_list) * 100, '% done\n')
		
		
		
	# Saving cortical depth response across Channel Map
	depth_shank = np.arange(electrode_spacing,electrode_spacing*(arr_chanMap.shape[0]+1),electrode_spacing)
	filename_save = 'Spectrogram-py-data' + '.mat'
	filename_save = os.path.join(output_dir_cortical_depth,filename_save)
	data_summary = {'empty':[0],'depth_shank':depth_shank,'MUA_depth_peak':MUA_depth_peak,'MUA_depth_mean':MUA_depth_mean,'LFP_depth_peak':LFP_depth_peak,'LFP_depth_mean':LFP_depth_mean,'LFP_depth_mean_post':LFP_depth_mean_post}
	savemat(filename_save,data_summary)
# fg2, (a1,a2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[1,1]})
# a1.set_ylabel('Max of ' + r'$\Delta$' + r'$P_n$')
# a2.set_ylabel('Mean of ' + r'$\Delta$' + r'$P_n$')
# a2.set_xlabel('Cortical Depth (' + r'$\mu m$' + ')')
# title_str = 'Cortical depth analysis of BC- 200ms stimulation (Shank-A)'
# fg2.suptitle(title_str)
# a1.set_xlim([depth_shank[0],depth_shank[-1]])
# a2.set_xlim([depth_shank[0],depth_shank[-1]])
# a1.plot(depth_shank,MUA_depth_peak[:,0],'r', lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,0,0],'k',lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,0,1],'y',lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,0,2],'b',lw = 1.75)
# a2.plot(depth_shank,MUA_depth_mean[:,0],'r', lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,0,0],'k',lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,0,1],'y',lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,0,2],'b',lw = 1.75)
# a2.legend(['MUA (0.3-3 KHz)',r'$\beta$',r'$\gamma$','LFP-high'],fontsize = 6)
# filename_save = 'depth analysis Shank-A' + '.png'
# filename_save = os.path.join(output_dir_cortical_depth,filename_save)
# fg.set_size_inches((9, 6), forward=False)
# plt.savefig(filename_save,format = 'png')
# plt.close('all')
# plt.clf()
# plt.cla()

# fg2, (a1,a2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[1,1]})
# a1.set_ylabel('Max of ' + r'$\Delta$' + r'$P_n$')
# a2.set_ylabel('Mean of ' + r'$\Delta$' + r'$P_n$')
# a2.set_xlabel('Cortical Depth (' + r'$\mu m$' + ')')
# title_str = 'Cortical depth analysis of BC- 200ms stimulation (Shank-B)'
# fg2.suptitle(title_str)
# a1.set_xlim([depth_shank[0],depth_shank[-1]])
# a2.set_xlim([depth_shank[0],depth_shank[-1]])
# a1.plot(depth_shank,MUA_depth_peak[:,1],'r', lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,1,0],'k',lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,1,1],'y',lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,1,2],'b',lw = 1.75)
# a2.plot(depth_shank,MUA_depth_mean[:,1],'r', lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,1,0],'k',lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,1,1],'y',lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,1,2],'b',lw = 1.75)
# a2.legend(['MUA (0.3-3 KHz)',r'$\beta$',r'$\gamma$','LFP-high'],fontsize = 6)
# filename_save = 'depth analysis Shank-B' + '.png'
# filename_save = os.path.join(output_dir_cortical_depth,filename_save)
# fg.set_size_inches((9, 6), forward=False)
# plt.savefig(filename_save,format = 'png')
# plt.close('all')
# plt.clf()
# plt.cla()

# fg2, (a1,a2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[1,1]})
# a1.set_ylabel('Max of ' + r'$\Delta$' + r'$P_n$')
# a2.set_ylabel('Mean of ' + r'$\Delta$' + r'$P_n$')
# a2.set_xlabel('Cortical Depth (' + r'$\mu m$' + ')')
# title_str = 'Cortical depth analysis of BC- 200ms stimulation (Shank-C)'
# fg2.suptitle(title_str)
# a1.set_xlim([depth_shank[0],depth_shank[-1]])
# a2.set_xlim([depth_shank[0],depth_shank[-1]])
# a1.plot(depth_shank,MUA_depth_peak[:,2],'r', lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,2,0],'k',lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,2,1],'y',lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,2,2],'b',lw = 1.75)
# a2.plot(depth_shank,MUA_depth_mean[:,2],'r', lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,2,0],'k',lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,2,1],'y',lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,2,2],'b',lw = 1.75)
# a2.legend(['MUA (0.3-3 KHz)',r'$\beta$',r'$\gamma$','LFP-high'],fontsize = 6)
# filename_save = 'depth analysis Shank-C' + '.png'
# filename_save = os.path.join(output_dir_cortical_depth,filename_save)
# fg.set_size_inches((9, 6), forward=False)
# plt.savefig(filename_save,format = 'png')
# plt.close('all')
# plt.clf()
# plt.cla()

# fg2, (a1,a2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[1,1]})
# a1.set_ylabel('Max of ' + r'$\Delta$' + r'$P_n$')
# a2.set_ylabel('Mean of ' + r'$\Delta$' + r'$P_n$')
# a2.set_xlabel('Cortical Depth (' + r'$\mu m$' + ')')
# title_str = 'Cortical depth analysis of BC- 200ms stimulation (Shank-D)'
# fg2.suptitle(title_str)
# a1.set_xlim([depth_shank[0],depth_shank[-1]])
# a2.set_xlim([depth_shank[0],depth_shank[-1]])
# a1.plot(depth_shank,MUA_depth_peak[:,3],'r', lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,3,0],'k',lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,3,1],'y',lw = 1.75)
# a1.plot(depth_shank,LFP_depth_peak[:,3,2],'b',lw = 1.75)
# a2.plot(depth_shank,MUA_depth_mean[:,3],'r', lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,3,0],'k',lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,3,1],'y',lw = 1.75)
# a2.plot(depth_shank,LFP_depth_mean[:,3,2],'b',lw = 1.75)
# a2.legend(['MUA (0.3-3 KHz)',r'$\beta$',r'$\gamma$','LFP-high'],fontsize = 6)
# filename_save = 'depth analysis Shank-D' + '.png'
# filename_save = os.path.join(output_dir_cortical_depth,filename_save)
# fg.set_size_inches((9, 6), forward=False)
# plt.savefig(filename_save,format = 'png')
# plt.close('all')
# plt.clf()
# plt.cla()

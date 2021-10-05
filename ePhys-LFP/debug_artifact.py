#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:36:40 2021

@author: hyr2
"""
import os, sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
from Support import *
dir_expsummary = '/run/media/hyr2/Data/Data/FH-BC6/2021-09-28/ePhys/whisker4/exp_summary.xlsx'
dir_Bin2Trials = '/run/media/hyr2/Data/Data/FH-BC6/2021-09-28/ePhys/whisker4/Bin2Trials'


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


# First channel only 
filename = os.path.join(dir_Bin2Trials,'Chan0.csv')
df_chan = pd.read_csv(filename,dtype = np.single)
arr_chan = df_chan.to_numpy()
Time = arr_chan[:,-1]
ADC_data = arr_chan[:,-2]
total_time = len(Time)               # in samples


# 0, 4
# 2, 3
y_1 = arr_chan[:,0]

eEEG_filtered = filterSignal_notch(y_1,Fs,60, axis_value = 0)
eEEG_MUA = filterSignal_MUA(y_1,Fs, axis_value = 0)
eEEG_filtered = filterSignal_lowpassLFP(y_1,Fs, axis_value=0)

# plt.figure(1,[10,20])
# plt.plot(Time,y_1)
# plt.figure(2,[10,20])
# plt.plot(Time,eEEG_filtered)
f1, a1 = plt.subplots(1, gridspec_kw={'height_ratios': [10]})
# plt.figure(4,[10,20])
a1.plot(Time,eEEG_MUA)
a1.set_xlim([0.5,6])


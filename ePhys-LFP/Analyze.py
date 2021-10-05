# -*- coding: utf-8 -*-
"""
Created on Thu May 20 01:19:54 2021

@author: Haad-Rathore
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from natsort import natsorted
from Support import *

source_dir = input('Enter the directory containing the output of Bin2Trials.py: \n')
source_dir_list = natsorted(os.listdir(source_dir))

# Extracting data from summary file .xlsx
Fs = 20e3

# Extracting data from .csv files
filename = os.path.join(source_dir,source_dir_list[62])
df = pd.read_csv(filename,dtype = np.single)
arr = df.to_numpy()
Trial1 = arr[:,15]
ADC_data = arr[:,-2]        # ADC data
Time = arr[:,-1]            # Timing data

# Filtering Data here
Trial1_filtered = filterSignal_LFP_Gamma_FIR(Trial1,Fs)
Trial1_filtered = filterSignal_Notch(Trial1_filtered,Fs)
ShowFFT(Trial1,Fs)
ShowFFT(Trial1_filtered,Fs)


plt.figure()
plt.plot(Time,Trial1_filtered)
plt.show()

plt.figure()
plt.plot(Time,Trial1,'r')
plt.plot(Time,Trial1_filtered+200,'g')
plt.plot(Time,9*ADC_data-200,'k',linewidth = 2)
plt.xlabel('Time(s)')
plt.ylabel('EEG')
plt.show()
plt.vlines(0.575,-600,1000,linestyles = 'dashed', colors = 'g')
plt.vlines(5.49,-600,1000,linestyles = 'dashed',colors = 'r')


# Python script to extract timing info from ePhys recording

import sys, os
sys.path.append(os.path.join(os.getcwd(),'Intan-File-Reader'))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import shutil
from natsort import natsorted
from load_intan_rhd_format import read_data
from Support import append_df_to_excel, read_stimtxt
from matplotlib import pyplot as plt
from scipy.io import savemat      # Import function to read data.

# Source directory
# |
# |__data*.rhd
# |__data*.rhd
# |__    .
# |__    .
# |__    .
# |__whisker_stim.txt

# Files and folders
print('Warning: Make sure "whisker_stim.txt" and "chan_map_1x32_128ch.xlsx" files are present in the current directory? \n')
# Raw_dir = input('Enter the raw directory folder:\n')
Raw_dir = '/home/hyr2-office/Documents/Data/temp_data/'
filename_trials_export = os.path.join(Raw_dir,'trials_times.mat')
source_dir_list = natsorted(os.listdir(Raw_dir))
matlabTXT = source_dir_list[source_dir_list.index('whisker_stim.txt')]
matlabTXT = os.path.join(Raw_dir,matlabTXT)

# Read .txt file
stim_start_time, stim_num, seq_period, len_trials, num_trials, FramePerSeq, total_seq, len_trials_arr = read_stimtxt(matlabTXT)

# Reading first file
Raw_dir_list = natsorted(os.listdir(Raw_dir))
filename = os.path.join(Raw_dir, Raw_dir_list[0])
df_final = pd.DataFrame(columns=['Time','ADC'])
for filename in Raw_dir_list:
    if filename.endswith('.rhd'):
        # Reading file
        filename = os.path.join(Raw_dir, filename)
        result = read_data(filename)
        # Writing timing and arr_ADC data
        # arr_ADC = result['board_adc_data']                        # Analog Trigger input from the CMOS 
        arr_ADC = result['board_dig_in_data']                       # Digital Trigger input 
        Time = result['t_amplifier']                        		# Timing info from INTAN
        arr_ADC = np.reshape(arr_ADC,(arr_ADC.size,))
        df = {'Time':Time,'ADC':arr_ADC}
        df = pd.DataFrame(df,dtype = np.single)
        df_final = pd.concat([df_final,df],axis = 0,ignore_index=True)
        Fs = result['frequency_parameters']['board_adc_sample_rate']    
        del result, df


arr_Time = pd.Series(df_final.Time)          # Time in seconds
arr_Time = arr_Time.to_numpy(dtype = np.single)
arr_ADC = pd.Series(df_final.ADC)            # ADC input (CMOS trigger)
arr_ADC = arr_ADC.to_numpy(dtype = np.single)
arr_ADC[arr_ADC >= 1] = 5                # Ceiling the ADC data (ideal signal)
arr_ADC[arr_ADC < 1] = 0                # Flooring the ADC data (ideal signal)
# Finding peaks
arr_ADC_diff = np.diff(arr_ADC)
arr_ADC_diff[arr_ADC_diff<0] = 0
arr_Time_diff = np.delete(arr_Time,[-1])
timestamp_frame = ( arr_ADC_diff - np.roll(arr_ADC_diff,1) > 0.5) & (arr_ADC_diff - np.roll(arr_ADC_diff,-1) > 0.5) # for digital
# Here I compute the indices of the timestamps 
timestamp_frame = timestamp_frame.nonzero()[0]                                        # Timestamp indices of the frames (FOIL Camera)
# sequences
temp_vec = np.diff(timestamp_frame)
x = np.argwhere(temp_vec > Fs*0.03)                                                   # Detect sequences
x = x.astype(int)
x = np.reshape(x,(len(x),))
x+=1
x = np.insert(x,0,0)                                                                  # So that we dont miss the first seq
timestamp_seq = timestamp_frame[x]
# trials
xx = np.argwhere(temp_vec > Fs*1)                                                      # Detect trials
xx = xx.astype(int)
xx = np.reshape(xx,(len(xx),))
xx+=1
xx = np.insert(xx,0,0)    

# xx = np.delete(xx,-1)   # extra

timestamp_trials = timestamp_frame[xx]

# Actual timestamps of the sequences and trials
timestamp_seq_times = arr_Time[timestamp_seq]           # in seconds
timestamp_trials_times = arr_Time[timestamp_trials]     # in seconds

#----------------------------- Plotting ---------------------------------------
plt.figure()
plt.plot(arr_Time,arr_ADC)
plt.plot(timestamp_seq_times,arr_ADC[timestamp_frame[x]]+1,'ro')
plt.plot(timestamp_trials_times,arr_ADC[timestamp_frame[xx]]+1,'go')
plt.show()

# Exporting Timestamps of the trial start times:
tt_export = timestamp_frame[xx]
export_timestamps_trials = {'empty':[0],'t_trial_start':tt_export}
savemat(filename_trials_export,export_timestamps_trials)
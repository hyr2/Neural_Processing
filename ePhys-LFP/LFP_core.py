# -*- coding: utf-8 -*-

import os, gc, warnings, json, sys, glob, shutil, pywt, pickle
from copy import deepcopy
sys.path.append(os.path.join(os.getcwd(),'Intan-File-Reader'))
sys.path.append(os.getcwd())
# from load_intan_rhd_format import Support
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from load_intan_rhd_format import read_data, get_n_samples_in_data
from Support import *
# from utils.mdaio import DiskWriteMda
# from utils.write_mda import writemda16i
# from utils.filtering import notch_filter
from natsort import natsorted
from matplotlib import pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000

# Reading general recording info
CMR = 1
decimate_factor = 60



source_dir = '/home/hyr2-office/Documents/Data/LFP/rh8/22-12-03/'
chanmap_filepath = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_flex_Pavlo.mat'

# Looking for important files in the source_dir
file_pre_MS = os.path.join(source_dir,'pre_MS.json')
file_trial_times = os.path.join(source_dir,'trials_times.mat')
file_trial_mask = os.path.join(source_dir,'trial_mask.csv')


files_rhd_list = glob.glob(os.path.join(source_dir,'*.rhd'))
source_dir_list = natsorted(files_rhd_list)

with open(file_pre_MS, 'r') as f:
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
N_trial_length = int(total_time * Fs)
dt = 1/Fs

if not CHMAP2X16:
    # 32x1 per shank setting
    GW_BETWEEN_SHANK = 300 # micron
    GH = 25 # micron
else:
    # 16x2 per shank setting, MirroHippo Fei Old Device ChMap
    GW_BETWEEN_SHANK = 250
    GW_WITHIN_SHANK = 30
    GH = 30

#########
CMR = int(CMR)
total_running_samples = []
for filename in source_dir_list:
    n_ch, n_samples_this_file = get_n_samples_in_data(filename)
    total_running_samples.append( n_samples_this_file)
    print(n_ch,'\t',n_samples_this_file)


# shank info and channel mapping


EEG_final = np.zeros([sum(total_running_samples),n_ch],dtype = float)
total_running_samples = np.cumsum(total_running_samples)
total_running_samples = np.insert(total_running_samples,0,0)

for iter1,filename in enumerate(source_dir_list):
    # Reading file
    result = read_data(filename)
    time_local = result['t_amplifier']                        # Timing info from INTAN
    eeg_local = result['amplifier_data']
    eeg_local = np.transpose(np.subtract(eeg_local,np.median(eeg_local, axis = 0)))
    indx_st = total_running_samples[iter1]
    indx_end = total_running_samples[iter1+1]
    EEG_final[indx_st:indx_end,:] = eeg_local

del eeg_local
# Channel mapping and reading recording channels
chan_list = []
for iter1 in range(n_ch):
    chan_list.append(result['amplifier_channels'][iter1]['native_order'])
chan_list = np.array(chan_list,dtype = np.int16)
chanmap = loadmat(chanmap_filepath)
chanmap = chanmap['Ch_Map_new']
if np.amin(chanmap) == 1:
    print("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
    chanmap -= 1
    

    
trial_times = loadmat(file_trial_times)
trial_times = trial_times['t_trial_start']
trial_times = np.squeeze(trial_times)
trial_mask = pd.read_csv(file_trial_mask,header = None).to_numpy()
trial_mask = np.array(np.squeeze(trial_mask),dtype = bool)
trial_times = trial_times[trial_mask]

# These are indices for the recording array EEG_final_trial
indx_shankA = []
indx_shankB = []
indx_shankC = []
indx_shankD = []
for iter1 in range(len(chan_list)):
    if np.where( chanmap ==  chan_list[iter1] ) [1] == 0:
        indx_shankA.append(iter1)
    if np.where( chanmap ==  chan_list[iter1] ) [1] == 1:
        indx_shankB.append(iter1)
    if np.where( chanmap ==  chan_list[iter1] ) [1] == 2:
        indx_shankC.append(iter1)
    if np.where( chanmap ==  chan_list[iter1] ) [1] == 3:
        indx_shankD.append(iter1)
        
indx_shankA = np.array(indx_shankA,dtype = np.int16)
indx_shankB = np.array(indx_shankB,dtype = np.int16)
indx_shankC = np.array(indx_shankC,dtype = np.int16)
indx_shankD = np.array(indx_shankD,dtype = np.int16)

# processing one shank at a time due to memory issues
# Splitting into trials 
EEG_final_trial = np.zeros([trial_mask.sum(),N_trial_length,indx_shankD.shape[0]],dtype = float)
for iter1 in range(trial_mask.sum()):
    EEG_final_trial[iter1,:,:] = EEG_final[trial_times[iter1]:trial_times[iter1]+N_trial_length,indx_shankD]
del EEG_final

# Filtering the signals (MUAs + Gamma Bands)
eEEG_filtered = filterSignal_notch(EEG_final_trial,Fs,60, axis_value = 1)
eEEG_filtered = filterSignal_Gamma(eEEG_filtered,Fs, axis_value = 1)
eEEG_filtered = signal.decimate(eEEG_filtered, decimate_factor, ftype = 'iir', axis = 1)

# Compute PSD 
cwt_arr_G, freq = pywt.cwt(eEEG_filtered,np.arange(10,100,1),'cmor0.2-2.0',sampling_period = decimate_factor/Fs, method = 'fft', axis = 1)
indx_freq_gamma = [(freq > 30) & (freq<100)]
indx_freq_gamma = np.array(indx_freq_gamma,dtype = bool)
indx_freq_gamma = np.squeeze(indx_freq_gamma)
cwt_arr_G = cwt_arr_G[indx_freq_gamma,:,:,:]
cwt_arr_G = np.mean(cwt_arr_G,axis = 0)     # Avg PSD in the Gamma band

# Time axis 
Fs_eff = Fs/decimate_factor
N_trial_length_new = N_trial_length/decimate_factor 
n_stim_start = n_stim_start/N_trial_length * N_trial_length_new
time_axis = np.arange(0, total_time, 1/Fs_eff)

# Baseline Normalized
t_baseline_cwt = np.where(np.logical_and(time_axis >= (stim_start_time-1.5), time_axis <= stim_start_time-0.1) )
t_baseline_cwt = np.asarray(t_baseline_cwt)
t_baseline_cwt = np.reshape(t_baseline_cwt,(t_baseline_cwt.size,))

# PSD
psd = np.square(np.abs(cwt_arr_G))    # PSD 
psd_bsl = np.mean(psd[:,t_baseline_cwt,:], axis = 1)    # avg baseline psd
psd = psd.transpose(1, 0, 2)
delta_PSD = (psd - psd_bsl)/psd_bsl

avg_trial = np.mean(delta_PSD,axis = 1)

dict_data = {
    'psd': avg_trial,
    'stim_start': n_stim_start,
    'time_axis': time_axis
    }

# avg_trial = np.square(np.abs(avg_trial))    # PSD 


# plot/save
filename_save = '/home/hyr2-office/Documents/Data/LFP/rh8/22-12-03/'
# plt.plot(avg_trial[:,0])
# plt.vlines(n_stim_start,0,1000)
# Save the dictionary to a binary file
filename_save = os.path.join(filename_save,'shankD.pkl')
with open(filename_save, "wb") as pickle_file:
    pickle.dump(dict_data, pickle_file)
# np.savez(os.path.join(filename_save,'avg_trial-singleshank.npz'),data1 = avg_trial,data2 = n_stim_start,data3 = time_axis)    # data1 data2 data3



# # ---------------------- Amplifier Data Binning ----------------------------

# # Names for columns for the exported .csv file
# list_titles = []
# for i in range(num_trials):
#     list_titles.append('Trial' + str(i+1))
# list_titles.append('ADC Trigger')
# list_titles.append('Time')
# # Reading each channels file and binning into trials 
# # This will create "Num_chan" .csv files
# iter_chan = 1
# del Data_dir_list[-1]
# for file in Data_dir_list:
#     filename_csv = Data_dir_list[iter_chan-1]             # Name of file will be exactly the same as the name of the original file
#     filename_csv = os.path.join(output_dir,filename_csv)
#     df_ChanData = pd.read_csv(os.path.join(Data_dir,file),skiprows = skip_opticalDelay, dtype = np.single)
#     arr_ChanData = df_ChanData.to_numpy()
#     arr_ChanData = arr_ChanData[:,0]
    
#     for iter_trial in range(num_trials):
#         start_index = int(timestamp_trials[iter_trial] - extra_time)                  # start index (start of trial - 0.5 sec)
#         # end_index = int(timestamp_seq[(iter_trial+1)*num_seq-1] + extra_time)       
#         end_index = int(start_index + smallest_duration_trial + 2*extra_time)         # end index   (end of trial + 0.5 sec)
#         trial_arr[:,iter_trial] = arr_ChanData[start_index:end_index]
#     df_Chan = pd.DataFrame(trial_arr, columns = list_titles)
#     df_Chan.to_csv(filename_csv, mode = 'w', header = True, index = False)
#     print('Channel data has been exported to: ',filename_csv)
#     print(iter_chan/len(Data_dir_list) * 100,'% done')
#     iter_chan += 1
    

# # Export exp_summary.xlsx : 
# data_summary = {'Sequence Time(s)':[stat_median_seq_time],'Stimulation Start Time(s)':[stim_start_t + extra_time/Fs]}
# df = pd.DataFrame(data_summary)
# append_df_to_excel(exp_summary_dir,df, sheet_name='Sheet1', startrow=4, index = False)          # appends into excel files conveniently    
    
# try:
#     shutil.rmtree(Data_dir)
# except OSError as e:
#     print("Error: %s - %s." % (e.filename, e.strerror))
    
    
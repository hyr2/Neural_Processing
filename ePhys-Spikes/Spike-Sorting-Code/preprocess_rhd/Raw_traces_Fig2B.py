#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:49:20 2024
Code will be used to generate Fig2B Raw traces
@author: hyr2-office
"""
from load_intan_rhd_format import read_data, read_header, get_n_samples_in_data, read_stimtxt
# from utils.mdaio import DiskWriteMda
# from utils.write_mda import writemda16i
# from utils.filtering import notch_filter
import os, time
from natsort import natsorted
from copy import deepcopy
from scipy.io import loadmat
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import signal

# Filters
def filterSignal_MUA(input_signal,Fs, axis_value = 0):
    # Prep
    signal_out = np.empty((input_signal.shape),dtype=np.single)

    cutoff = np.array([500,3000])
    sos = signal.butter(10, cutoff, btype = 'bandpass', output = 'sos', fs = Fs)  # IIR filter
    signal_out = signal.sosfiltfilt(sos, input_signal, axis = axis_value)
    return signal_out
def filterSignal_LFP(input_signal,Fs, axis_value = 0):
    # Prep
    signal_out = np.empty((input_signal.shape),dtype=np.single)

    cutoff = np.array([30,80])
    sos = signal.butter(5, cutoff, btype = 'bandpass', output = 'sos', fs = Fs)  # IIR filter
    signal_out = signal.sosfiltfilt(sos, input_signal, axis = axis_value)
    return signal_out


# output_foldeer
t = time.localtime()
current_time = time.strftime("%m_%d_%Y_%H_%M", t)
current_time = current_time + '_Fig2B'
# output_folder = os.path.join(r'C:\Rice_2023\Data\Results',current_time)
output_folder = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loading data (RH11 23-07-19)
Raw_dir = '/home/hyr2-office/Documents/Data/Raw_ePhys_Trace'    
chan_map_path = os.path.join(Raw_dir,'chan_map_1x32_128ch_flex_Pavlo.mat')
filenames = os.listdir(Raw_dir)
filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
filenames = list(filter(lambda x: "rh11" in x, filenames))    
filenames = natsorted(filenames)
data_dict = read_data(os.path.join(Raw_dir, filenames[0]))

chs_info = deepcopy(data_dict['amplifier_channels'])
arr_ADC = data_dict['board_dig_in_data']                       # Digital Trigger input 
Time = data_dict['t_amplifier']                        		# Timing info from INTAN
Fs = data_dict['sample_rate']

# Channel mappings
chs_native_order_local = [e['native_order'] for e in chs_info]
chmap_mat = loadmat(chan_map_path)['Ch_Map_new']
if np.min(chmap_mat)==1:
    print("    Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
    chmap_mat -= 1
df = pd.DataFrame(data = None,columns = ['Shank','Depth'])
lst_r_id = []
lst_sh_id = []
for iter_localA in range(len(chs_native_order_local)):
    loc_local = np.squeeze(np.argwhere(chmap_mat == chs_native_order_local[iter_localA]))
    r_id = loc_local[0]
    sh_id = loc_local[1]
    lst_r_id.append(r_id)
    lst_sh_id.append(sh_id)
df['Shank'] = lst_sh_id 
df['Depth'] = lst_r_id

# Selecting single shank and all rows
ephys_data = data_dict['amplifier_data']
df_tmp = df.loc[df['Shank'] == 0]
indx_accepted = deepcopy(list(df_tmp.index))
df_tmp['indexx'] = indx_accepted
df_tmp = df_tmp.sort_values(by = 'Depth', ascending = True,axis = 0)
selected_indx = df_tmp['indexx'].to_numpy()
ephys_data = ephys_data[selected_indx,:]
for iter_i in range(len(selected_indx)):
    ephys_data[iter_i,:] = ephys_data[iter_i,:] + 250*(iter_i+1)
fig,ax = plt.subplots(1,1,figsize = (4,6))
plt.plot(ephys_data[:,9000:12000].T,color = '#4d4e4d',linewidth = 1.5)
sns.despine(top = True, bottom =True,left = True)
ax.set_xticks([])
ax.set_yticks([])
filename_save = os.path.join(output_folder,'Fig2B_0.png')
plt.savefig(filename_save,dpi=300,format='png')

# Selecting single shank and all rows
ephys_data = data_dict['amplifier_data']
df_tmp = df.loc[df['Shank'] == 1]
indx_accepted = deepcopy(list(df_tmp.index))
df_tmp['indexx'] = indx_accepted
df_tmp = df_tmp.sort_values(by = 'Depth', ascending = True,axis = 0)
selected_indx = df_tmp['indexx'].to_numpy()
ephys_data = ephys_data[selected_indx,:]
for iter_i in range(len(selected_indx)):
    ephys_data[iter_i,:] = ephys_data[iter_i,:] + 250*(iter_i+1)
fig,ax = plt.subplots(1,1,figsize = (4,6))
ax.plot(ephys_data[:,9000:12000].T,color = '#4d4e4d',linewidth = 1.5)
sns.despine(top = True, bottom =True,left = True)
ax.set_xticks([])
ax.set_yticks([])
filename_save = os.path.join(output_folder,'Fig2B_1.png')
plt.savefig(filename_save,dpi=300,format='png')

# Selecting single shank and all rows
ephys_data = data_dict['amplifier_data']
df_tmp = df.loc[df['Shank'] == 2]
indx_accepted = deepcopy(list(df_tmp.index))
df_tmp['indexx'] = indx_accepted
df_tmp = df_tmp.sort_values(by = 'Depth', ascending = True,axis = 0)
selected_indx = df_tmp['indexx'].to_numpy()
ephys_data = ephys_data[selected_indx,:]
for iter_i in range(len(selected_indx)):
    ephys_data[iter_i,:] = ephys_data[iter_i,:] - 200*(iter_i+1)
fig,ax = plt.subplots(1,1,figsize = (6,8.5))
ax.plot(ephys_data[:,12000:15000].T,color = '#4d4e4d',linewidth = 1)
sns.despine(top = True, bottom =True,left = True)
ax.set_xticks([])
ax.set_yticks([])
filename_save = os.path.join(output_folder,'Fig2B_2.png')
plt.savefig(filename_save,dpi=300,format='png')

# Selecting single shank and all rows
ephys_data = data_dict['amplifier_data']
df_tmp = df.loc[df['Shank'] == 3]
indx_accepted = deepcopy(list(df_tmp.index))
df_tmp['indexx'] = indx_accepted
df_tmp = df_tmp.sort_values(by = 'Depth', ascending = True,axis = 0)
selected_indx = df_tmp['indexx'].to_numpy()
ephys_data = ephys_data[selected_indx,:]
for iter_i in range(len(selected_indx)):
    ephys_data[iter_i,:] = ephys_data[iter_i,:] + 250*(iter_i+1)
fig,ax = plt.subplots(1,1,figsize = (4,6))
ax.plot(ephys_data[:,1143932:1151462].T,color = '#4d4e4d',linewidth = 1)
sns.despine(top = True, bottom =True,left = True)
ax.set_xticks([])
ax.set_yticks([])
filename_save = os.path.join(output_folder,'Fig2B_3.png')
plt.savefig(filename_save,dpi=300,format='png')


# For Figure 3C
# Selecting single shank and all rows
Os = 50000
ephys_data = data_dict['amplifier_data']
df_tmp = df.loc[df['Shank'] == 3]
indx_accepted = deepcopy(list(df_tmp.index))
df_tmp['indexx'] = indx_accepted
df_tmp = df_tmp.sort_values(by = 'Depth', ascending = True,axis = 0)
selected_indx = df_tmp['indexx'].to_numpy()
ephys_data = ephys_data[selected_indx,:]
for iter_i in range(len(selected_indx)):
    ephys_data[iter_i,:] = ephys_data[iter_i,:] + 250*(iter_i+1)
fig,ax = plt.subplots(1,1,figsize = (4,6))
yy = ephys_data[0:3,1143900:1151499]
yy = filterSignal_MUA(yy, Fs,axis_value=1)
for iter_i in range(yy.shape[0]):
    yy[iter_i,:] = yy[iter_i,:] + 650*(iter_i+1)
ax.plot(yy.T,color = '#4d4e4d',linewidth = 1.5)
sns.despine(top = True, bottom =True,left = True)
ax.set_xticks([])
ax.set_yticks([])
filename_save = os.path.join(output_folder,'Fig3C_3.png')
plt.savefig(filename_save,dpi=300,format='png')


# EXTRA NOT FOR PAPER

# Selecting single shank and all rows
ephys_data = data_dict['amplifier_data']
df_tmp = df.loc[df['Shank'] == 0]
indx_accepted = deepcopy(list(df_tmp.index))
df_tmp['indexx'] = indx_accepted
df_tmp = df_tmp.sort_values(by = 'Depth', ascending = True,axis = 0)
selected_indx = df_tmp['indexx'].to_numpy()
ephys_data = ephys_data[selected_indx,:]
for iter_i in range(3):
    ephys_data[iter_i,:] = ephys_data[iter_i,:] + 250*(iter_i+1)
fig,ax = plt.subplots(1,1,figsize = (4,6))
plt.plot(ephys_data[0:3,9000:12000].T,color = '#4d4e4d',linewidth = 1.5)
sns.despine(top = True, bottom =True,left = True)
ax.set_xticks([])
ax.set_yticks([])
filename_save = os.path.join(output_folder,'RAW_TRACE_EXTRA_0.png')
plt.savefig(filename_save,dpi=300,format='png')


ephys_1 = ephys_data[0:3,9000:12000]
ephys_lfp = filterSignal_LFP(ephys_1.T, Fs)
ephys_mua = filterSignal_MUA(ephys_1.T, Fs)

ephys_lfp = ephys_lfp.T
for iter_i in range(3):
    ephys_lfp[iter_i,:] = ephys_lfp[iter_i,:] + 250*(iter_i+1)
fig,ax = plt.subplots(1,1,figsize = (4,6))
plt.plot(ephys_lfp.T,color = '#4d4e4d',linewidth = 1.5)
sns.despine(top = True, bottom =True,left = True)
ax.set_xticks([])
ax.set_yticks([])
filename_save = os.path.join(output_folder,'RAW_TRACE_EXTRA_lfp.png')
plt.savefig(filename_save,dpi=300,format='png')


ephys_mua = ephys_mua.T
for iter_i in range(3):
    ephys_mua[iter_i,:] = ephys_mua[iter_i,:] + 250*(iter_i+1)
fig,ax = plt.subplots(1,1,figsize = (4,6))
plt.plot(ephys_mua.T,color = '#4d4e4d',linewidth = 1.5)
sns.despine(top = True, bottom =True,left = True)
ax.set_xticks([])
ax.set_yticks([])
filename_save = os.path.join(output_folder,'RAW_TRACE_EXTRA_mua.png')
plt.savefig(filename_save,dpi=300,format='png')









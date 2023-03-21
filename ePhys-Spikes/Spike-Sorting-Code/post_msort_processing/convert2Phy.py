#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:57:09 2023

@author: hyr2-office
"""

# This standalone script is used to convert the curated output files from 
# discard_noise_viz.py file and give the correct output files to be used for
# manual curation in the software PHY

import os, sys, json
sys.path.append(os.path.join(os.getcwd(),'utils-mountainsort'))
sys.path.append(os.getcwd())
from itertools import groupby
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy.signal as signal
import pandas as pd
from utils.Support import read_stimtxt
from utils.read_mda import readmda

session_folder = '/home/hyr2-office/Documents/Data/NVC/RH-7-merging-orig/10-17-22/'
output_phy = os.path.join(session_folder,'phy_output')
if not os.path.exists(output_phy):
    os.makedirs(output_phy)


# Everyting except PCA (feature space)
firings = readmda(os.path.join(session_folder, "firings.mda")).astype(np.int64)
template_waveforms = readmda(os.path.join(session_folder, "templates.mda")).astype(np.float64)


nSpikes = firings.shape[1]  # number of spikes in the sessions

prim_ch = firings[0,:]   
spike_times = firings[1,:].astype(dtype = 'uint64')         # This is spike_times.npy
spike_clusters = firings[2,:].astype(dtype = 'uint32') - 1  # This is spike_clusters.npy

nTemplates = np.max(spike_clusters) + 1                      # number of clusters found
nChan = np.amax(prim_ch)                                  # number of channels in the recording (effective)

# Needs to be fixed (read all_waveforms_by_cluster.npz)
amplitudes = np.zeros([nSpikes,])
all_waveforms_by_cluster = np.load(os.path.join(session_folder,'all_waveforms_by_cluster.npz'))
for i_clus in range(nTemplates):
    waveforms_this_cluster = all_waveforms_by_cluster['clus%d'%(i_clus+1)]  # cluster IDs in .npz start from 1
    waveform_peaks = np.max(waveforms_this_cluster, axis=1)
    waveform_troughs = np.min(waveforms_this_cluster, axis=1)
    tmp_amp_series = (waveform_peaks-waveform_troughs) * (1-2*(waveform_peaks<0))
    indx_amplitudes = np.where(spike_clusters == i_clus)[0]     # spike_clusters starts from 0
    # assert tmp_amp_series.shape[0] == indx_amplitudes.shape[0], \
    if (tmp_amp_series.shape[0] != indx_amplitudes.shape[0]):       # @jiaaoz please take a look at this error
        print('Number of spikes in cluster %d did not match'%(i_clus+1))
        tmp_num = indx_amplitudes.shape[0] - tmp_amp_series.shape[0] 
        for iter in range(tmp_num):
            tmp_amp_series = np.append(tmp_amp_series,np.mean(tmp_amp_series))
        # print(tmp_amp_series.shape[0] - indx_amplitudes.shape[0])
    amplitudes[indx_amplitudes] = tmp_amp_series
    
amplitudes = amplitudes.astype(dtype = 'float64')
# amplitudes = np.ones([nSpikes,],dtype = 'float64')         # This is amplitudes.npy

template_waveforms = np.moveaxis(template_waveforms,[0,1,2],[2,1,0]).astype(dtype = 'float32') # This is templates.npy

spike_templates = spike_clusters                         # This is spike_templates.npy

templates_ind = np.ones([nTemplates,nChan],dtype = 'float64')       
tmp_arr = np.arange(0,nChan,1)
templates_ind = templates_ind * np.transpose(tmp_arr)    # This is templates_ind.npy

channel_map = tmp_arr                                    # This is channel_map.npy 
channel_positions = pd.read_csv(os.path.join(session_folder,'geom.csv'), header=None).values
channel_positions = channel_positions.astype(dtype = 'float64') # This is channel_positions.npy

# Creating the params.py

file_pre_ms = os.path.join(session_folder,'pre_MS.json')
with open(file_pre_ms, 'r') as f:
  data_pre_ms = json.load(f)
F_SAMPLE = float(data_pre_ms['SampleRate'])

fname = os.path.join(output_phy,'params.py')
path_data = "'empty str'"
l1 = 'dat_path = ' + path_data + '\n'
l2 = 'n_channels_dat = ' + str(nChan) + '\n'
l3 = 'dtype = ' + "'int16'" + '\n'
l4 = 'offset = ' + str(0) + '\n'
l5 = 'sample_rate = ' + str(F_SAMPLE) + '\n'
l6 = 'hp_filtered = ' + str('False') + '\n'

with open(fname, 'w') as f:
    f.writelines([l1,l2,l3,l4,l5,l6])

# Applying the curation mask (accept_mask.csv)
curation_mask_file = os.path.join(session_folder,'accept_mask.csv')
curation_mask = pd.read_csv(curation_mask_file, header=None, index_col=False,dtype = bool)
curation_mask_np = curation_mask.to_numpy()
curation_mask_np = np.reshape(curation_mask_np,[curation_mask_np.shape[0],])
tmp = curation_mask_np[spike_clusters]    # remove these spikes from the firings.mda structure
spike_clusters_new = deepcopy(spike_clusters[tmp])
# spike_templates_new = deepcopy(spike_templates[tmp])
spike_times_new = deepcopy(spike_times[tmp])
amplitudes_new = deepcopy(amplitudes[tmp])
templates_ind_new = deepcopy(templates_ind[curation_mask_np,:])
template_waveforms_new = deepcopy(template_waveforms[curation_mask_np,:,:])
nTemplates_new = np.sum(curation_mask_np)
new_cluster_id = np.arange(0,nTemplates_new,1)
old_cluster_id = np.unique(spike_clusters_new)
# cluster_mapping = old_cluster_id
for iter in range(nTemplates_new):
    indx_temp = np.where(spike_clusters_new == old_cluster_id[iter])[0]
    spike_clusters_new[indx_temp] = iter
spike_templates_new = spike_clusters_new
cluster_mapping = np.vstack((new_cluster_id,old_cluster_id))
cluster_mapping = np.transpose(cluster_mapping)
pd.DataFrame(data=cluster_mapping.astype(int)).to_csv(os.path.join(output_phy, "cluster_mapping.csv"), index=False, header=False)
# spike_templates_new = spike_clusters_new

# Generating similar_templates.npy containing possible merging candidates (Jiaao's code)


np.save(os.path.join(output_phy,'spike_times.npy'),spike_times_new)
np.save(os.path.join(output_phy,'spike_clusters.npy'),spike_clusters_new)
np.save(os.path.join(output_phy,'amplitudes.npy'),amplitudes_new)
np.save(os.path.join(output_phy,'templates.npy'),template_waveforms_new)
np.save(os.path.join(output_phy,'spike_templates.npy'),spike_templates_new)
np.save(os.path.join(output_phy,'templates_ind.npy'),templates_ind_new)
np.save(os.path.join(output_phy,'channel_map.npy'),channel_map)
np.save(os.path.join(output_phy,'channel_positions.npy'),channel_positions)

# For feature space (PCA)

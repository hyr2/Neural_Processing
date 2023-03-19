#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:57:09 2023

@author: hyr2-office
"""

import os, sys, json
sys.path.append(os.path.join(os.getcwd(),'utils-mountainsort'))
sys.path.append(os.getcwd())
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy.signal as signal
import pandas as pd
from utils.Support import read_stimtxt
from utils.read_mda import readmda

session_folder = '/home/hyr2-office/Documents/Data/NVC/RH-7-merging/10-24-22/'
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

np.save(os.path.join(output_phy,'spike_times.npy'),spike_times)
np.save(os.path.join(output_phy,'spike_clusters.npy'),spike_clusters)
np.save(os.path.join(output_phy,'amplitudes.npy'),amplitudes)
np.save(os.path.join(output_phy,'templates.npy'),template_waveforms)
np.save(os.path.join(output_phy,'spike_templates.npy'),spike_templates)
np.save(os.path.join(output_phy,'templates_ind.npy'),templates_ind)
np.save(os.path.join(output_phy,'channel_map.npy'),channel_map)
np.save(os.path.join(output_phy,'channel_positions.npy'),channel_positions)


# For feature space (PCA)

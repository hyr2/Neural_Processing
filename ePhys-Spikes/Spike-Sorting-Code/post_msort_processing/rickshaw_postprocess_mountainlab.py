#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:30:52 2022

"""

# This script is used to perform post processing following spike sorting. It performs automatic curation of the clusters 
# (putative neurons) as well as population analysis of stimulus locked clusters (ie the firing rates). 
# Make sure the working directory is where this script resides.
import sys
sys.path.append(r'../../Time-Series/')
import numpy as np
from discard_noise_and_viz import *
from population_analysis import *
from FR_TS_pop import combine_sessions 
import os, json
from natsort import natsorted
import matlab.engine

# Automate batch processing of the pre processing step

# read parameters
with open("../params.json", "r") as f:
    params = json.load(f)

input_dir = params['spikesort_dir']
CHANNEL_MAP_FPATH = params['CHANNEL_MAP_FPATH']
source_dir_list = natsorted(os.listdir(input_dir))

# Start matlab engine and change directory to code file
eng = matlab.engine.start_matlab()
eng.cd(r'../../Connectivity Analysis/', nargout=0)

# Iterate over all sessions
for iter, filename in enumerate(source_dir_list):
    print(iter, ' ',filename)
    Raw_dir = os.path.join(input_dir, filename)
    if os.path.isdir(Raw_dir):

        file_pre_ms = os.path.join(Raw_dir,'pre_MS.json')
        with open(file_pre_ms, 'r') as f:
            data_pre_ms = json.load(f)
        F_SAMPLE = np.float(data_pre_ms['SampleRate'])

        # Curation
        # func_discard_noise_and_viz(Raw_dir)
        # Population analysis
        # func_pop_analysis(Raw_dir,CHANNEL_MAP_FPATH)

        # Calling matlab scripts from python
        eng.func_CE_BarrelCortex(Raw_dir,F_SAMPLE,CHANNEL_MAP_FPATH,nargout=0)

        # delete converted_data.mda and filt.mda and raw data files (.rhd) 
        os.remove(os.path.join(Raw_dir,'converted_data.mda')) if os.path.isfile(os.path.join(Raw_dir,'converted_data.mda')) else None
        os.remove(os.path.join(Raw_dir,'filt.mda')) if os.path.isfile(os.path.join(Raw_dir,'filt.mda')) else None
        test = os.listdir(Raw_dir)
        for item in test:
            if item.endswith(".rhd"):
                os.remove(os.path.join(Raw_dir, item))

eng.quit()

# Running FR_TS_pop.py script for complete statics of this mouse over all sessions (longitudinal)
# WARNING: Make sure to edit script for the correct animal (since their measurement days might differ)
combine_sessions(input_dir,'Mouse_ID')
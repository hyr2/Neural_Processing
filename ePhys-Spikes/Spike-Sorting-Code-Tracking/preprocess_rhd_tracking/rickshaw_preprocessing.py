#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 5 20:30:52 2023

@author: HR
"""
# Automate batch processing of the pre processing step (shank wise sorting is performed from all sessions together)
# This script will be useful for single neuronal tracking
from preprocess_rhd import func_preprocess
import os, subprocess, json
from natsort import natsorted

# This script is used to perform preprocessing before the spike sorting. Additionally it also performs spike sorting by calling
# the mountainsort bash script. It requires mountainlab (conda environment) to be installed.
# In addition the working directory should where this script exists (i.e. `preprocess_rhd/`).

# Raw_dir
#         |
#         |__11-02/
#         |__11-03/
#         |__11-04/
#         |__shanks_.json
# Each session folder must have all the .rhd files and the whisker_stim.txt file

# read parameters
with open("../params.json", "r") as f:
    params = json.load(f)

# Folder location inputs
input_dir = params['raw_dir']
output_dir = params['spikesort_dir']
CHANNEL_MAP_FPATH = params['CHANNEL_MAP_FPATH']
# Mountain Sort inputs
ELECTRODE_2X16 = params['ELECTRODE_2X16']
num_features_var = params['msort_num_features']
max_num_clips_for_pca_var = params["msort_max_num_clips_for_pca"]
ml_temp_dir = params["msort_temp_dir"]

# Pre-processing step
# func_preprocess(Raw_dir, output_dir, ELECTRODE_2X16, CHANNEL_MAP_FPATH)

source_dir_list = natsorted(os.listdir(output_dir))  # folders are shanks
for iter, filename in enumerate(source_dir_list):
    print(iter, ' ', filename)
    Raw_dir = os.path.join(output_dir, filename)    # single shank directory
    output_dir_MS = Raw_dir
    if os.path.isdir(Raw_dir):
      # Preparing for MountainSort script
      # Reading from .json file
      file_pre_ms = os.path.join(Raw_dir,'pre_MS.json')
      with open(file_pre_ms, 'r') as f:
        data_pre_ms = json.load(f)
      F_SAMPLE = str(data_pre_ms['SampleRate'])
      # geom.csv 
      geom_filepath = os.path.join(Raw_dir,'geom.csv')
      
      # Calling MountainSort Bash script
      subprocess.call(['bash','./mountainSort128_stroke_hyr2.sh',output_dir_MS,output_dir_MS,F_SAMPLE,geom_filepath,num_features_var,max_num_clips_for_pca_var,ml_temp_dir])


# shanks_.json
# {
#     "A": true,
#     "B": true,
#     "C": false,
#     "D": true
# }
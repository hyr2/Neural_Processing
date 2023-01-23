#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:30:52 2022

@author: luanlab
"""
# Automate batch processing of the pre processing step
from preprocess_rhd import *
import os, subprocess, json
from natsort import natsorted

# This script is used to perform preprocessing before the spike sorting. Additionally it also performs spike sorting by calling
# the mountainsort bash script. It requires mountainlab (conda environment) to be installed.
# In addition the working directory should where this script exists (i.e. `preprocess_rhd/`).

# input_dir
#         |
#         |__11-02/
#         |__11-03/
#         |__11-04/
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

source_dir_list = natsorted(os.listdir(input_dir))
for iter, filename in enumerate(source_dir_list):
    print(iter, ' ', filename )
    Raw_dir = os.path.join(input_dir, filename)
    output_dir_MS = os.path.join(output_dir, filename)
    if os.path.isdir(Raw_dir):
      # Running preprocessing step
      func_preprocess(Raw_dir, output_dir, ELECTRODE_2X16, CHANNEL_MAP_FPATH)

      # Preparing for MountainSort script
      # Reading from .json file
      file_pre_ms = os.path.join(output_dir_MS,'pre_MS.json')
      with open(file_pre_ms, 'r') as f:
        data_pre_ms = json.load(f)
      F_SAMPLE = str(data_pre_ms['SampleRate'])
      # geom.csv 
      geom_filepath = os.path.join(output_dir_MS,'geom.csv')
      
      # Calling MountainSort Bash script
      subprocess.call(['bash','./mountainSort128_stroke_hyr2.sh',output_dir_MS,output_dir_MS,F_SAMPLE,geom_filepath,num_features_var,max_num_clips_for_pca_var,ml_temp_dir])
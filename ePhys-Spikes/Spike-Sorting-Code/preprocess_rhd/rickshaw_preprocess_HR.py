#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:30:52 2022

@author: luanlab
"""
# Automate batch processing of the pre processing step
from preprocess_rhd_HR import *
import os, subprocess, json
from natsort import natsorted

# input_dir
#         |
#         |__11-02/
#         |__11-03/
#         |__11-04/
# Each session folder must have all the .rhd files and the whisker_stim.txt file

# Folder location inputs
input_dir = '/home/hyr2-office/Documents/Data/NVC/RH-7/'
output_dir = '/home/hyr2-office/Documents/Data/NVC/RH-7/'
CHANNEL_MAP_FPATH = "/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_2x16_flex_rk18.mat"

# Mountain Sort inputs
ELECTRODE_2X16 = True
num_features_var = "8"
max_num_clips_for_pca_var = "1000"

source_dir_list = natsorted(os.listdir(input_dir))
for iter, filename in enumerate(source_dir_list):
    print(iter, ' ', filename )
    Raw_dir = os.path.join(input_dir, filename)
    output_dir_MS = os.path.join(output_dir, filename)

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
    subprocess.call(['bash','./mountainSort128_stroke_hyr2.sh',output_dir_MS,output_dir_MS,F_SAMPLE,geom_filepath,num_features_var,max_num_clips_for_pca_var])
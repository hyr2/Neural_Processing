#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:30:52 2022

"""

# This script is used to perform post processing following spike sorting. It performs automatic curation of the clusters 
# (putative neurons) as well as population analysis of stimulus locked clusters (ie the firing rates).It also performs cell type analysis,
# and excitatory vs inhibitory cell population trakcing post stroke. It calls cell explorer (matlab).
# This is a complete, fully automated script that is doing about a dozen of various spike analysis
# plots being generated for our longitudinal ephys data. 
# Make sure the working directory is where this script resides.
# After this script runs (without error), use the phy_output contents to perform manual merging of oversplit clusters

import numpy as np
from discard_noise_and_viz import *
import os, json, sys
from natsort import natsorted
sys.path.append(r'../../Time-Series/')
from FR_TS_pop import *
from convert2Phy import func_convert2Phy, func_convert2MS
from apply_curation_and_save import clean_mdas_main
# Automate batch processing of the pre processing step


if __name__ == '__main__':

    # PHY manual curation flag
    phy_flag = 1                        # indicates that PHY manual curation has been performed (we use it only for merging of clusters)

    # read parameters
    with open("../params.json", "r") as f:
        params = json.load(f)

    input_dir = params['spikesort_dir']
    CHANNEL_MAP_FPATH = params['CHANNEL_MAP_FPATH']
    source_dir_list = natsorted(os.listdir(input_dir))
    mda_reldirs = []

    # Iterate over all sessions
    for iter, filename in enumerate(source_dir_list):
        print(iter, ' ',filename)
        Raw_dir = os.path.join(input_dir, filename)
        if os.path.isdir(Raw_dir):
            mda_reldirs.append(Raw_dir)
            file_pre_ms = os.path.join(Raw_dir,'pre_MS.json')
            with open(file_pre_ms, 'r') as f:
                data_pre_ms = json.load(f)
            F_SAMPLE = float(data_pre_ms['SampleRate'])
                            
            # Curation
            func_discard_noise_and_viz(Raw_dir)
            
            # delete converted_data.mda and filt.mda and raw data files (.rhd) 
            # os.remove(os.path.join(Raw_dir,'converted_data.mda'))
            # os.remove(os.path.join(Raw_dir,'filt.mda'))
            # test = os.listdir(Raw_dir)
            # for item in test:
            #     if item.endswith(".rhd"):
            #         os.remove(os.path.join(Raw_dir, item))

    # Applying curation masks to generate "*_clean" files
    clean_mdas_main(mda_reldirs,mda_reldirs)                       #(script is called clean_firings_mda.py written by JiaaoZ)

    # Iterate over all sessions to perform conversion of data into PHY compatible files
    for iter, filename in enumerate(source_dir_list):
        print(iter, ' ',filename)
        Raw_dir = os.path.join(input_dir, filename)
        if os.path.isdir(Raw_dir):
            func_convert2Phy(Raw_dir)
        
# At the end of this code, you must perform manual curation using PHY ----- *******************                
        
        

# Sample Params.json file for reference:
#     {
#     "raw_dir": "/home/hyr2-office/Documents/Data/NVC/B-BC5/",
#     "spikesort_dir": "/home/hyr2-office/Documents/Data/NVC/B-BC5/",
#     "CHANNEL_MAP_FPATH": "/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_2x16_rigid_hanlin_fei.mat",
#     "ELECTRODE_2X16": true,
#     "msort_num_features": "8",
#     "msort_max_num_clips_for_pca": "1000",
#     "msort_temp_dir": "/home/hyr2-office/Documents/Data/NVC/ml-temp"
# }

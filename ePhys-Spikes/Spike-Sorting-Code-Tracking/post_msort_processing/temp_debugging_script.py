#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:00:41 2023

@author: hyr2-office
"""
import numpy as np
from discard_noise_and_viz import *
from population_analysis import *
import os, json, sys
from natsort import natsorted
import matlab.engine
from convert2Phy import func_convert2Phy, func_convert2MS
from combine_shanks_single_animal import combine_shanks
from ACG_compute_matlab_wrap import func_acg_extract_main

# from convert2Phy import func_convert2Phy, func_convert2MS

Raw_dir = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_0'
# CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_flex_new.mat'
# input_dir = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_bc7/'
# session_folder = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_bc7/'
# input_dir = session_folder
# func_convert2MS(session_folder)

# func_pop_analysis(Raw_dir,CHANNEL_MAP_FPATH)
func_acg_extract_main(Raw_dir)
# combine_shanks(session_folder)
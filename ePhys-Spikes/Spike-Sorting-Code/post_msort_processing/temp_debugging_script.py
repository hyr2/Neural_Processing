#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:00:41 2023

@author: hyr2-office
"""
import sys
sys.path.append(r'../../Time-Series/')
import numpy as np
from discard_noise_and_viz import *
from population_analysis import *
from FR_TS_pop import *
import os, json
from natsort import natsorted

# from convert2Phy import func_convert2Phy, func_convert2MS

# Raw_dir = '/home/hyr2-office/Documents/Data/NVC/RH-7_REJECTED/10-27-22/'
CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_flex_new.mat'
# input_dir = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_bc7/'
session_folder = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_bbc5/'
input_dir = session_folder
# func_convert2MS(session_folder)

# func_pop_analysis(session_folder,CHANNEL_MAP_FPATH)
# func_discard_noise_and_viz(Raw_dir)

# func_pop_analysis(Raw_dir,CHANNEL_MAP_FPATH)
moouse_id = input_dir.split('/')[-2]
combine_sessions(input_dir,moouse_id)
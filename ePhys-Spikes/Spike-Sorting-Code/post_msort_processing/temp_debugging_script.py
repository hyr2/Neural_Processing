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

Raw_dir = '/home/hyr2-office/Documents/Data/NVC/RH-8/22-12-3/'
CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_2x16_flex_Pavlo.mat'
input_dir = '/home/hyr2-office/Documents/Data/NVC/RH-9/'


moouse_id = input_dir.split('/')[-2]

# func_pop_analysis(Raw_dir,CHANNEL_MAP_FPATH)
combine_sessions(input_dir,moouse_id)
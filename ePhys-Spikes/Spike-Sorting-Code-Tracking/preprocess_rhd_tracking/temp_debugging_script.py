#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:39:07 2023

@author: hyr2-office
"""

from preprocess_rhd import func_preprocess
import os

Raw_dir = '/home/hyr2-office/Documents/Data/NVC/Tracking/rh11'
output_dir = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11'
ELECTRODE_2X16 = False
CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_flex_Pavlo.mat'

func_preprocess(Raw_dir, output_dir, ELECTRODE_2X16, CHANNEL_MAP_FPATH)
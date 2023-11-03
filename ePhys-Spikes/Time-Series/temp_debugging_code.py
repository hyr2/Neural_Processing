#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:30:10 2023

@author: hyr2-office
"""

from min_chan_map_standalone import find_min_chan_map

electrode_2x16 = False
input_dir = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh11'
channel_map_path = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_flex_Pavlo.mat'

find_min_chan_map(input_dir,channel_map_path,electrode_2x16)
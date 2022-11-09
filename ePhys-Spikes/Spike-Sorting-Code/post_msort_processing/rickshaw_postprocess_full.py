#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:30:52 2022

@author: luanlab
"""

from discard_noise_and_viz_HR import *
from population_analysis import *
import os
from natsort import natsorted
# Automate batch processing of the pre processing step

input_dir = '/home/hyr2-office/Documents/Data/NVC/BC6/'
CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_rigid.mat'
source_dir_list = natsorted(os.listdir(input_dir))

for iter, filename in enumerate(source_dir_list):
    print(iter, ' ', filename )
    Raw_dir = os.path.join(input_dir, filename)
    if os.path.isdir(Raw_dir):
        # func_discard_noise_and_viz(Raw_dir)
        func_pop_analysis(Raw_dir,CHANNEL_MAP_FPATH)

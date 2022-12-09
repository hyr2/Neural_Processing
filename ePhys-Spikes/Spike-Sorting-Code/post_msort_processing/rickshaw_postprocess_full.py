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

input_dir = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/Haad/temp_RY'
CHANNEL_MAP_FPATH = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat'
source_dir_list = natsorted(os.listdir(input_dir))

for iter, filename in enumerate(source_dir_list):
    print(iter, ' ', filename )
    Raw_dir = os.path.join(input_dir, filename)
    func_discard_noise_and_viz(Raw_dir)
    # func_pop_analysis(Raw_dir,CHANNEL_MAP_FPATH)

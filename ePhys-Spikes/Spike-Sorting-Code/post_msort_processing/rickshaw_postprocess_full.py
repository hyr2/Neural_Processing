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

input_dir = '/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/B-BC5'
CHANNEL_MAP_FPATH = '/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/Neural_Processing/Channel_Maps/Mirro_Oversampling_hippo_map.mat'
source_dir_list = natsorted(os.listdir(input_dir))

for iter, filename in enumerate(source_dir_list):
    print(iter, ' ',filename)
    Raw_dir = os.path.join(input_dir, filename)
    if os.path.isdir(Raw_dir):
        func_discard_noise_and_viz(Raw_dir)
        # func_pop_analysis(Raw_dir,CHANNEL_MAP_FPATH)
        # delete converted_data.mda and filt.mda
        # os.remove(os.path.join(Raw_dir,'converted_data.mda'))
        # os.remove(os.path.join(Raw_dir,'filt.mda'))



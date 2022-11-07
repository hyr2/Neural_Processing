#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:30:52 2022

@author: luanlab
"""

from preprocess_rhd_HR import *
import os
# Automate batch processing of the pre processing step




input_dir = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/HR/bc7'
output_dir = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/Haad/bc7/'
CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat"
ELECTRODE_2X16 = False
source_dir_list = natsorted(os.listdir(input_dir))

for iter, filename in enumerate(source_dir_list):
    print(iter, ' ', filename )
    Raw_dir = os.path.join(input_dir, filename)
    func_preprocess(Raw_dir, output_dir, ELECTRODE_2X16, CHANNEL_MAP_FPATH)

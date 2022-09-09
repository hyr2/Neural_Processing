#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 04:11:37 2021

@author: hyr2
"""

# this script is used to rename the files

import os
from natsort import natsorted
import shutil
import numpy as np
import pandas as pd

source_dir = input('Enter the input directory:\n')
source_dir_list= natsorted(os.listdir(source_dir))

chan_list_path = input('Path to channel list:\n')
df_chanList = pd.read_excel(chan_list_path,header = 0)
chan_list = df_chanList.to_numpy()
chan_list = np.reshape(chan_list,(len(source_dir_list),))

output_folder = os.path.join(source_dir,'Renamed')
os.mkdir(output_folder)

for iter in range(len(source_dir_list)):
    filename_in = source_dir_list[iter]
    filename_out = 'Chan' + str(chan_list[iter]) + '.csv'
    shutil.copyfile(os.path.join(source_dir,filename_in),os.path.join(output_folder,filename_out))
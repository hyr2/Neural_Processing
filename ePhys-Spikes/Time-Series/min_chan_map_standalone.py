# This is a standalone script used to generate the minimum channel map mask 
# for a single animal (all sessions). Obviously it needs one single channel 
# map file and the input_dir : animal directory 

# this is the native channel order (ie starts from 0)

# It requires already processed data (preprocessing script must be run (rickshaw_preprocess.py))

# input_dir
#     |
#     |____10-22-22
#     |____10-23-22
#     |____    .
#     |____    .
#     |____    .

import scipy.io as sio # Import function to read data.
import numpy as np
import os, sys
from natsort import natsorted
from copy import deepcopy
sys.path.append(r'../Spike-Sorting-Code/post_msort_processing/')
import pandas as pd
from utils.read_mda import readmda
from utils.read_stimtxt import read_stimtxt
from functools import reduce


# Finding the minimum channel map (only for 2x16 so far)

ELECTRODE_2X16 = True
input_dir = '/home/hyr2-office/Documents/Data/NVC/RH-8/'
channel_map_path = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_2x16_flex_Pavlo.mat'

# # channel spacing 
if not ELECTRODE_2X16:
    GW_BETWEEN_SHANK = 300 # micron
    GH = 25 # micron
else:
    # 16x2 per shank setting, MirroHippo Fei Old Device ChMap
    GW_BETWEEN_SHANK = 250
    GW_WITHIN_SHANK = 30
    GH = 30

min_chanmap_mask_list = []
# min_chanmap_mask_list1 = []
source_dir_list = natsorted(os.listdir(input_dir))
for iter, filename in enumerate(source_dir_list):
    session_folder = os.path.join(input_dir,filename)
    if os.path.isdir(session_folder):
        file_path_geom = os.path.join(session_folder,'geom.csv')
        file_path_nativemap = os.path.join(session_folder,'native_ch_order.npy')
        geom = pd.read_csv(file_path_geom)
        geom_np = geom.to_numpy()
    
        # firings = readmda(os.path.join(session_folder, "firings.mda")).astype(np.int64)
        chan_map = sio.loadmat(channel_map_path)['Ch_Map_new']
        chan_map = np.squeeze(chan_map)
        if np.min(chan_map)==1:
            print("    Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
            chan_map -= 1
        
        if not ELECTRODE_2X16:
            print('Channel map is 1x32\n')
        else:
            print('Channel map is 2x16\n')
            loc = np.zeros([16,8])
            for i, (x,y)  in enumerate(zip(geom_np[:,0],geom_np[:,1])):
                shank_ID_loc1 = x // GW_BETWEEN_SHANK
                shank_ID_loc2 = ((x - (shank_ID_loc1 * GW_BETWEEN_SHANK))//GW_WITHIN_SHANK)%2
                col_ID = 2*shank_ID_loc1 + shank_ID_loc2
                row_ID = y//GH
    
                loc[row_ID,col_ID] = 1  # binary mask for channels to be accepted
                
        # np.save(os.path.join(session_folder,'min_chanmap_mask.npy'),loc)
        
        # andd = np.logical_and(chan_map ,loc)
        # linear_arr_min = chan_map[andd]
        # linear_arr_min = deepcopy(np.sort(linear_arr_min,kind = 'mergesort'))
        # min_chanmap_mask_list.append(linear_arr_min)
        min_chanmap_mask_list.append(np.load(file_path_nativemap))
        
final_out = reduce(np.intersect1d,(min_chanmap_mask_list[iter] for iter in range(len(min_chanmap_mask_list))))
np.save(os.path.join(input_dir,'min_chanmap_mask.npy'),final_out)   # this is the native channel order (ie starts from 0)




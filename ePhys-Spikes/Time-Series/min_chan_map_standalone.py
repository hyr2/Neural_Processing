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
from matplotlib import pyplot as plt


def gen_geommap(chmap_mat,list_channels):
    # Takes channel map and and takes list of channel IDs (native) and generates
    # the geom.csv type numpy array. Currently for 2x16
    geom_map = -1*np.ones((len(list_channels), 2), dtype=np.int16)
    for i, native_order in enumerate(list_channels):
        loc = np.where(chmap_mat==native_order)
        geom_map[i,0] = (loc[1][0]//2)*GW_BETWEEN_SHANK + (loc[1][0]%2)*GW_WITHIN_SHANK
        geom_map[i,1] = loc[0][0]*GH
    return geom_map

def gen_1x16(geom_np): 
    loc = np.zeros([16,4])   # this is the modified version of the 2x16 layout
    for i, (x,y)  in enumerate(zip(geom_np[:,0],geom_np[:,1])):
        shank_ID_loc1 = x // GW_BETWEEN_SHANK
        # shank_ID_loc2 = ((x - (shank_ID_loc1 * GW_BETWEEN_SHANK))//GW_WITHIN_SHANK)%2
        # col_ID = 2*shank_ID_loc1 + shank_ID_loc2
        row_ID = y//GH
        loc[row_ID,shank_ID_loc1] = 1  # binary mask for channels to be accepted
    
    return loc

# Finding the minimum channel map (only for 2x16 so far)

ELECTRODE_2X16 = True
input_dir = '/home/hyr2-office/Documents/Data/NVC/RH-9/'
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
min_chanmap_mask_list_new = []
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
            loc = gen_1x16(geom_np)
            
            # loc = np.zeros([16,8])
            # for i, (x,y)  in enumerate(zip(geom_np[:,0],geom_np[:,1])):
            #     shank_ID_loc1 = x // GW_BETWEEN_SHANK
            #     shank_ID_loc2 = ((x - (shank_ID_loc1 * GW_BETWEEN_SHANK))//GW_WITHIN_SHANK)%2
            #     col_ID = 2*shank_ID_loc1 + shank_ID_loc2
            #     row_ID = y//GH
            #     loc[row_ID,col_ID] = 1  # binary mask for channels to be accepted
                
        # np.save(os.path.join(session_folder,'min_chanmap_mask.npy'),loc)
        min_chanmap_mask_list_new.append(loc)       # for method 3 (see onenote: https://rice-my.sharepoint.com/personal/hyr2_rice_edu/_layouts/15/Doc.aspx?sourcedoc={54e662f5-5165-40d0-9c26-41f7943d4993}&action=edit&wd=target%28Paper.one%7Ca7689642-f93d-4492-af25-88732fe0739f%2FChannel%20normalizations%7Cce6aeaa4-aa55-483e-a1dc-90474bd0285a%2F%29&wdorigin=NavigationUrl)
        # andd = np.logical_and(chan_map ,loc)
        # linear_arr_min = chan_map[andd]
        # linear_arr_min = deepcopy(np.sort(linear_arr_min,kind = 'mergesort'))
        # min_chanmap_mask_list.append(linear_arr_min)
        min_chanmap_mask_list.append(np.load(file_path_nativemap)) # for method 4 (seee onenote: https://rice-my.sharepoint.com/personal/hyr2_rice_edu/_layouts/15/Doc.aspx?sourcedoc={54e662f5-5165-40d0-9c26-41f7943d4993}&action=edit&wd=target%28Paper.one%7Ca7689642-f93d-4492-af25-88732fe0739f%2FChannel%20normalizations%7Cce6aeaa4-aa55-483e-a1dc-90474bd0285a%2F%29&wdorigin=NavigationUrl)
        
final_out = reduce(np.intersect1d,(min_chanmap_mask_list[iter] for iter in range(len(min_chanmap_mask_list))))

geom_map_orig = gen_geommap(chan_map,final_out)
for iter in range(len(min_chanmap_mask_list)):
    tmp_str = "chan_comp_method3_{}".format(iter) +  '.png'
    filename_save = os.path.join(input_dir,tmp_str)
    f, a = plt.subplots(1,1)
    a.plot(geom_map_orig[:,0],400-geom_map_orig[:,1],'o')
    a.plot(gen_geommap(chan_map,min_chanmap_mask_list[iter])[:,0],400-gen_geommap(chan_map,min_chanmap_mask_list[iter])[:,1],'x')
    f.set_size_inches((10, 6), forward=False)
    plt.savefig(filename_save,format = 'png')
    plt.close(f)
    
final_out_new = reduce(np.logical_and,(min_chanmap_mask_list_new[iter] for iter in range(len(min_chanmap_mask_list_new))))
geom_map_orig = np.argwhere(final_out_new == 1)
for iter in range(len(min_chanmap_mask_list_new)):
    tmp_str = "chan_comp_method4_{}".format(iter) +  '.png'
    filename_save = os.path.join(input_dir,tmp_str)
    f, a = plt.subplots(1,1)
    geom_map_current = np.argwhere(min_chanmap_mask_list_new[iter] == 1)
    a.scatter(geom_map_orig[:,1],-geom_map_orig[:,0],marker = 'o')
    a.scatter(geom_map_current[:,1],-geom_map_current[:,0],marker = 'x')
    f.set_size_inches((10, 6), forward=False)
    plt.savefig(filename_save,format = 'png')
    plt.close(f)


np.save(os.path.join(input_dir,'min_chanmap_mask.npy'),final_out)   # this is the native channel order (ie starts from 0) # This is for method 3 (see onenote)
np.save(os.path.join(input_dir,'min_chanmap_mask_new.npy'),final_out_new)   # This is a binary mask (1x16 effective layout for 2x16 device) # This is for method 4




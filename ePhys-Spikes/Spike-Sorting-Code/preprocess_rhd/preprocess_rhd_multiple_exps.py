'''
Preprocess .rhd files from multiple experiments into one .mda file
Only use channels that are commonly accepted in all experiments 
'''
#%%
import os
import gc
from random import sample
import warnings
from copy import deepcopy
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from load_intan_rhd_format import read_data, read_header, get_n_samples_in_data
from utils.mdaio import DiskWriteMda
from utils.write_mda import writemda16i
from utils.filtering import notch_filter
from natsort import natsorted

def find_ids_of_values(arr, target_values):
    '''find the ids of the target_values in array, such that arr[ids] == targe_values 
    [arr] and [target_values] are both 1-d arrays
    '''
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    ids = [np.where(arr==tv)[0][0] for tv in target_values]
    return np.array(ids)

# def test():
#     x= [7,9,8,11,2,3,4]
#     z1=[7,4,2]
#     z2=[7,2,4]
#     z3 = np.array(z1)
#     print(find_ids_of_values(x, z1))
#     print(x)
#     print(find_ids_of_values(x,z3))
#     print(x)
#     print(find_ids_of_values(x,z2))
#     print(x)
#     exit(0)

# channel map .mat file
# BC6, B-BC8 is rigid
# BC7, BC8 is flex
# B-BC5 is 2x16 (Fei old channelmap)
CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat" # 1x32 flex
# CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/chan_map_1x32_128ch_rigid.mat" # 1x32 rigid
# CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/Fei2x16old/Mirro_Oversampling_hippo_map.mat" # 2x16

ELECTRODE_2X16 = False

# channel spacing
# 32x1 per shank setting
if not ELECTRODE_2X16:
    GW_BETWEEN_SHANK = 300 # micron
    GH = 25 # micron
else:
    # 16x2 per shank setting, MirroHippo Fei Old Device ChMap
    GW_BETWEEN_SHANK = 250
    GW_WITHIN_SHANK = 30
    GH = 30


# given a session
DATA_ROOTPATH  = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/"
MDA_ROOTPATH   = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/converted/data_mda/"
GEOM_ROOTPATH  = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out"
SESSION_REL_PATH_DEST = "NVC/BC7/combined0"
SESSION_REL_PATHS_SRC = [
    "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/NVC/BC7/12-06-2021",
    "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/NVC/BC7/12-07-2021",
    "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/NVC/BC7/12-09-2021",
    "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/NVC/BC7/01-07-2022",
    ]

SESSION_FOLDER_MDA = os.path.join(MDA_ROOTPATH, SESSION_REL_PATH_DEST)
SESSION_FOLDER_CSV = os.path.join(GEOM_ROOTPATH, SESSION_REL_PATH_DEST)



if not os.path.exists(SESSION_FOLDER_MDA):
    os.makedirs(SESSION_FOLDER_MDA)

if not os.path.exists(SESSION_FOLDER_CSV):
    os.makedirs(SESSION_FOLDER_CSV)


# get the commonly accepted channels and total number of samples
n_samples_by_exp = []
n_samples_cumsum_by_files_allexps = []
chs_native_orders = []

print("-"*20)
print("Scanning headers")
print("-"*20)
sample_freq = None
for session_rel_path in SESSION_REL_PATHS_SRC:
    session_folder_raw = os.path.join(DATA_ROOTPATH, session_rel_path)
    print()
    filenames = os.listdir(session_folder_raw)
    filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
    filenames = natsorted(filenames)
    # print(filenames)

    # get file #samples
    n_samples_cumsum_by_file = [0]
    n_samples = 0
    for filename in filenames:
        n_ch, n_samples_this_file = get_n_samples_in_data(os.path.join(session_folder_raw, filename))
        n_samples += n_samples_this_file
        n_samples_cumsum_by_file.append(n_samples)
    n_samples_by_exp.append(n_samples) # total number of samples in the experiment
    n_samples_cumsum_by_files_allexps.append(n_samples_cumsum_by_file)

    # get channel native orders
    with open(os.path.join(session_folder_raw, filename), "rb") as fh:
        head_dict = read_header(fh, verbose=False)
        chs_info = deepcopy(head_dict['amplifier_channels'])
        if sample_freq is None:
            sample_freq = head_dict['sample_rate']
        elif head_dict['sample_rate'] != sample_freq:
            raise ValueError("Previous sample freq is %d Hz, this experiment has %d Hz" % (sample_freq, head_dict['sample_rate']))
    chs_native_order_this_exp = [e['native_order'] for e in chs_info] # just a 1d python array
    chs_native_orders.append(chs_native_order_this_exp)

# find commonly accepted channels
native_orders_as_sets = [set(cnote) for cnote in chs_native_orders]
native_order_set = native_orders_as_sets[0].intersection(*native_orders_as_sets[1:])
chs_native_order_final = list(native_order_set)
print("# channels commonly accepted in all experiments", len(chs_native_order_final))
print("These are (in native orders):")
print(chs_native_order_final)

# now get the indices corresponding to the common native channels for each experiment
ch_select_ids = []
for i_exp, native_order_this_exp in enumerate(chs_native_orders):
    ch_select_ids.append(find_ids_of_values(native_order_this_exp, chs_native_order_final))

# get total n_samples and n_chs
n_samples_cumsum_by_exp = [0] + np.cumsum(n_samples_by_exp).tolist()
n_samples_in_total = np.sum(n_samples_by_exp)
n_ch_allexps = len(chs_native_order_final)



print("-"*20)
print("Reading data, fs=", sample_freq)
print("-"*20)
writer = DiskWriteMda(os.path.join(SESSION_FOLDER_MDA, "converted_data.mda"), (n_ch_allexps, n_samples_in_total), dt="int16")
for i_exp, session_rel_path in enumerate(SESSION_REL_PATHS_SRC):
    session_folder_raw = os.path.join(DATA_ROOTPATH, session_rel_path)
    filenames = os.listdir(session_folder_raw)
    filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
    filenames = natsorted(filenames)
    for i_file, filename in enumerate(filenames):
        print("----%s----"%(os.path.join(session_folder_raw, filename)))
        data_dict = read_data(os.path.join(session_folder_raw, filename))
        
        print("Data are read") # no need to notch since we only care about 250~5000Hz
        ephys_data = data_dict['amplifier_data']
        # ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
        ephys_data = ephys_data.astype(np.int16)
        ephys_data = ephys_data[ch_select_ids[i_exp], :] # select commonly accepted channels only
        print("Appending chunk to disk, shape=", ephys_data.shape)
        entry_offset = (n_samples_cumsum_by_exp[i_exp]+n_samples_cumsum_by_files_allexps[i_exp][i_file]) * n_ch_allexps
        writer.writeChunk(ephys_data, i1=0, i2=entry_offset)
        del(ephys_data)
        del(data_dict)
        gc.collect()



# generate geom.csv

# read .mat for channel map (make sure channel index starts from 0)
if not ELECTRODE_2X16:
    chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
    # print(list(chmap_mat.keys()))
    if np.min(chmap_mat)==1:
        print("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
        chmap_mat -= 1

    print(chmap_mat.shape)
    if chmap_mat.shape!=(32,4):
        raise ValueError("Channel map is of shape %s, expected (32,4)" % (chmap_mat.shape))

    # find correct locations for valid chanels
    geom_map = -1*np.ones((len(chs_native_order_final), 2), dtype=np.int)

    for i, native_order in enumerate(chs_native_order_final):
        loc = np.where(chmap_mat==native_order)
        geom_map[i,0] = loc[1][0]*GW_BETWEEN_SHANK
        geom_map[i,1] = loc[0][0]*GH
else:
    # chmap_mat = loadmat(CHANNEL_MAP_FPATH)["Maps"].squeeze().tolist()
    # chmap_mat = np.concatenate(chmap_mat, axis=1) # should be (16, 8)
    chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
    if np.min(chmap_mat)==1:
        print("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
        chmap_mat -= 1
    print(chmap_mat.shape)
    if chmap_mat.shape!=(16,8):
        raise ValueError("Channel map is of shape %s, expected (16,8)" % (chmap_mat.shape))
    geom_map = -1*np.ones((len(chs_native_order_final), 2), dtype=np.int)
    for i, native_order in enumerate(chs_native_order_final):
        loc = np.where(chmap_mat==native_order)
        geom_map[i,0] = (loc[1][0]//2)*GW_BETWEEN_SHANK + (loc[1][0]%2)*GW_WITHIN_SHANK
        geom_map[i,1] = loc[0][0]*GH

#%%


# generate .csv
geom_map_df = pd.DataFrame(data=geom_map)
geom_map_df.to_csv(os.path.join(SESSION_FOLDER_CSV, "geom.csv"), index=False, header=False)
print("Geom file generated")
infodict = {"sample_freq": sample_freq}
infodict['experiments'] = SESSION_REL_PATHS_SRC
infodict['n_samples_by_exp'] = n_samples_by_exp
# savemat(os.path.join(SESSION_FOLDER_CSV, "info.mat"), infodict)
with open(os.path.join(SESSION_FOLDER_CSV, "info.json"), "w") as fjson:
    json.dump(infodict, fjson)
np.save(os.path.join(SESSION_FOLDER_CSV, "native_ch_order.npy"), chs_native_order_final)
print("Done!")


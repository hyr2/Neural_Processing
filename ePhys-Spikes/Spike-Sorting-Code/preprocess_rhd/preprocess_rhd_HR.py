#%%
import os
import gc
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



# channel map .mat file
# BC6, B-BC8 is rigid
# BC7, BC8 is flex
# B-BC5 is 2x16 flex
CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/chan_map_1x32_128ch_rigid.mat" # rigid
# CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/Fei2x16old/Mirro_Oversampling_hippo_map.mat" # flex 
# CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/oversampling_palvo_flex_intanAdapterMap.mat"
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


#/media/luanlab/DATA/SpikeSorting/RawData/2021-09-04B-aged/2022-01-01/2022-01-01_moving

# given a session
DATA_ROOTPATH  = '/media/luanlab/DATA/SpikeSorting/RawData/2021-09-04B-aged'
# MDA_ROOTPATH   = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/converted/data_mda/Yifu"
GEOM_ROOTPATH  = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/Kai/'
SESSION_REL_PATH = '2021-12-17'
SESSION_FOLDER_RAW = os.path.join(DATA_ROOTPATH, SESSION_REL_PATH)
# SESSION_FOLDER_MDA = os.path.join(MDA_ROOTPATH, SESSION_REL_PATH)
SESSION_FOLDER_CSV = os.path.join(GEOM_ROOTPATH, SESSION_REL_PATH)
SESSION_FOLDER_MDA = SESSION_FOLDER_CSV

filenames = os.listdir(SESSION_FOLDER_RAW)
filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
filenames = natsorted(filenames)

print(filenames)

if not os.path.exists(SESSION_FOLDER_MDA):
    os.makedirs(SESSION_FOLDER_MDA)

if not os.path.exists(SESSION_FOLDER_CSV):
    os.makedirs(SESSION_FOLDER_CSV)
#%%
def check_header_consistency(hA, hB):
    if len(hA)!=len(hB): 
        return False
    for a, b in zip(hA, hB):
        if a!=b:
            return False
    return True

# read rhd files and append data in list 
### REMEMBER native order starts from 0 ###
n_samples_cumsum_by_file = [0]
n_samples = 0
for filename in filenames:
    n_ch, n_samples_this_file =get_n_samples_in_data(os.path.join(SESSION_FOLDER_RAW, filename))
    n_samples += n_samples_this_file
    n_samples_cumsum_by_file.append(n_samples) #count of total V points
    

writer = DiskWriteMda(os.path.join(SESSION_FOLDER_MDA, "converted_data.mda"), (n_ch, n_samples), dt="int16")

chs_impedance = None
notch_freq = None
data_rhd_list = []
ephys_data_whole = None
chs_native_order = None
sample_freq = None
for i_file, filename in enumerate(filenames):
    print("----%s----"%(os.path.join(SESSION_FOLDER_RAW, filename)))
    with open(os.path.join(SESSION_FOLDER_RAW, filename), "rb") as fh:
    
        head_dict = read_header(fh)
    # Saving sampling rate info for future use
    if i_file == 0:
        dictionary_summary = {
            "Session": filename[0:filename.index('_')],
            "NumChannels": head_dict['num_amplifier_channels'],
            "SampleRate": head_dict['sample_rate'],
            "ELECTRODE_2X16": ELECTRODE_2X16
        }
        # Serializing json
        json_object = json.dumps(dictionary_summary, indent=4)
        with open(os.path.join(SESSION_FOLDER_MDA,'pre_MS.json'), "w") as outfile:
            outfile.write(json_object)

    data_dict = read_data(os.path.join(SESSION_FOLDER_RAW, filename))
    chs_info = deepcopy(data_dict['amplifier_channels'])
    
    # record and check key information
    if chs_native_order is None:
        chs_native_order = [e['native_order'] for e in chs_info]
        chs_impedance = [e['electrode_impedance_magnitude'] for e in chs_info]
        print("#Chans with >= 3MOhm impedance:", np.sum(np.array(chs_impedance)>=3e6))
        notch_freq = head_dict['notch_filter_frequency']
        sample_freq = head_dict['sample_rate']
    else:
        tmp_native_order = [e['native_order'] for e in chs_info]
        print("#Chans with >= 3MOhm impedance:", np.sum(np.array([e['electrode_impedance_magnitude'] for e in chs_info])>=3e6))
        if not check_header_consistency(tmp_native_order, chs_native_order):
            warnings.warn("WARNING in preprocess_rhd: native ordering of channels inconsistent within one session\n")
        if notch_freq != head_dict['notch_filter_frequency']:
            warnings.warn("WARNING in preprocess_rhd: notch frequency inconsistent within one session\n")
        if sample_freq != head_dict['sample_rate']:
            warnings.warn("WARNING in preprocess_rhd: sampling frequency inconsistent within one session\n")
    
    # print("Applying notch")
    print("Data are read") # no need to notch since we only care about 250~5000Hz
    ephys_data = data_dict['amplifier_data']
    # ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
    ephys_data = ephys_data.astype(np.int16)
    print("Appending chunk to disk")
    entry_offset = n_samples_cumsum_by_file[i_file] * n_ch
    writer.writeChunk(ephys_data, i1=0, i2=entry_offset)
    del(ephys_data)
    del(data_dict)
    gc.collect()
    # print("Concatenating")
    # if ephys_data_whole is None:
    #     ephys_data_whole = ephys_data
    # else:
    #     ephys_data_whole = np.concatenate([ephys_data_whole, ephys_data], axis=1)
    #     del(ephys_data)
    #     del(data_dict)
    #     gc.collect()

# print("Saving mda...")
# ##save to mda
# writemda16i(ephys_data_whole, os.path.join(SESSION_FOLDER_MDA, "converted_data.mda"))
# print("MDA file saved to %s" % (os.path.join(SESSION_FOLDER_MDA, "converted_data.mda")))

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
    geom_map = -1*np.ones((len(chs_native_order), 2), dtype=np.int)

    for i, native_order in enumerate(chs_native_order):
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
    geom_map = -1*np.ones((len(chs_native_order), 2), dtype=np.int)
    for i, native_order in enumerate(chs_native_order):
        loc = np.where(chmap_mat==native_order)
        geom_map[i,0] = (loc[1][0]//2)*GW_BETWEEN_SHANK + (loc[1][0]%2)*GW_WITHIN_SHANK
        geom_map[i,1] = loc[0][0]*GH

#%%


# generate .csv
geom_map_df = pd.DataFrame(data=geom_map)
geom_map_df.to_csv(os.path.join(SESSION_FOLDER_CSV, "geom.csv"), index=False, header=False)
print("Geom file generated")
infodict = {"sample_freq": sample_freq}
# savemat(os.path.join(SESSION_FOLDER_CSV, "info.mat"), infodict)
with open(os.path.join(SESSION_FOLDER_CSV, "info.json"), "w") as fjson:
    json.dump(infodict, fjson)
np.save(os.path.join(SESSION_FOLDER_CSV, "native_ch_order.npy"), chs_native_order)
print("Done!")


# C:\Yifu\2021-09-04-aged\2021-12-15\2021-12-15_211215_221904
# C:\Yifu\2021-09-04-ageds\2021-12-15\2021-12-15_211215_221904
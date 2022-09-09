"""
Preprocess 4x32 intan rhd data block files into complete continuous mda files by shank as well as corresponding positions
Assumes the channel map ordering & impedance stays the same throughout session

"""
import os
import gc
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from load_intan_rhd_format import read_data, read_header
from utils.write_mda import writemda16i
from utils.filtering import notch_filter

# channel map .mat file
# BC6 is rigid
# BC7, BC8 is flex
# CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/chan_map_1x32_128ch_rigid.mat" # rigid
CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat" # flex 
# channel spacing
GW = 300 # micron
GH = 25 # micron

# given a session
DATA_ROOTPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/"
MDA_ROOTPATH  = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/converted/data_mda/"
GEOM_ROOTPATH  = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out"
SESSION_REL_PATH = "NVC/BC7/12-06-2021" # 07, 09(pre stroke), 12(post stroke)
SESSION_FOLDER_RAW = os.path.join(DATA_ROOTPATH, SESSION_REL_PATH)
SESSION_FOLDER_MDA = os.path.join(MDA_ROOTPATH, SESSION_REL_PATH)
SESSION_FOLDER_CSV = os.path.join(GEOM_ROOTPATH, SESSION_REL_PATH)

filenames = os.listdir(SESSION_FOLDER_RAW)
filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))

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
# data_rhd_list = []
def preprocess_byshank(chmap_mat, shank_map):
    

    ephys_data_whole = None
    chs_native_order = None
    chs_impedance = None
    notch_freq = None
    sample_freq = None
    for filename in filenames:
        print("----%s----"%(os.path.join(SESSION_FOLDER_RAW, filename)))
        with open(os.path.join(SESSION_FOLDER_RAW, filename), "rb") as fh:
            head_dict = read_header(fh)
        data_dict = read_data(os.path.join(SESSION_FOLDER_RAW, filename))
        chs_info = data_dict['amplifier_channels']
        
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
        
        print("Applying notch")
        ephys_data = data_dict['amplifier_data']
        ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
        print("Concatenating")
        if ephys_data_whole is None:
            ephys_data_whole = ephys_data
        else:
            ephys_data_whole = np.concatenate([ephys_data_whole, ephys_data], axis=1)
            del(ephys_data)
            del(data_dict)
            gc.collect()

    print("Saving mda...")
    # save to mda
    writemda16i(ephys_data_whole, os.path.join(SESSION_FOLDER_MDA, "converted_data.mda"))
    print("MDA file saved to %s" % (os.path.join(SESSION_FOLDER_MDA, "converted_data.mda")))

    # generate geom.csv

    

    # generate .csv
    geom_map_df = pd.DataFrame(data=geom_map)
    geom_map_df.to_csv(os.path.join(SESSION_FOLDER_CSV, "geom.csv"), index=False, header=False)
    print("Geom file generated")
    infodict = {"sample_freq": sample_freq}
    savemat(os.path.join(SESSION_FOLDER_CSV, "info.mat"), infodict)
print("Done!")
# %%
def preprocess_single_session():
    # read .mat for channel map (make sure channel index starts from 0)
    chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
    # print(list(chmap_mat.keys()))
    if np.min(chmap_mat)==1:
        print("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
        chmap_mat -= 1

    print(chmap_mat.shape)
    if chmap_mat.shape!=(32,4):
        raise ValueError("Channel map is of shape %s, expected (32,4)" % (chmap_mat.shape))

    for i in range(4):
        print("##########processing shank#%d/4"%(i+1))
        preprocess_byshank(chmap_mat, i)
    
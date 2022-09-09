#%%
import os

import numpy as np
from scipy.io import loadmat
import pandas as pd
from load_intan_rhd_format import read_header

# get valid channels with correct ordering from raw data file .rhd (native order starts from 0)
# Assumes troke project data storage format
SESSION_FOLDER = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/NVC/BC6/BC6-11-11-2021"
CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat" # rigid or flex 
TARGET_CSV_FPATH = "./tmp.csv"
# channel spacing
GW = 300 # micron
GH = 25 # micron
session_rhd_fnames = list(filter(lambda x: x.endswith(".rhd"), os.listdir(SESSION_FOLDER)))
session_rhd_fnames = list(map(lambda x: os.path.join(SESSION_FOLDER, x), session_rhd_fnames))
with open(session_rhd_fnames[0], "rb") as f:
    rhd_head = read_header(f)
valid_ch_impedance = [k['electrode_impedance_magnitude'] for k in rhd_head['amplifier_channels']]
valid_ch_native_order = [k['native_order'] for k in rhd_head['amplifier_channels']]

# read .mat for channel map (make sure channel index starts from 0)
chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
# print(list(chmap_mat.keys()))
if np.min(chmap_mat)==1:
    chmap_mat -= 1
print(chmap_mat.shape)
if chmap_mat.shape!=(32,4):
    raise ValueError("Channel map is of shape %s, expected (32,4)" % (chmap_mat.shape))

#%%
# find correct locations for valid chanels
geom_map = -1*np.ones((len(valid_ch_native_order), 2), dtype=np.int)
for i, native_order in enumerate(valid_ch_native_order):
    loc = np.where(chmap_mat==native_order)
    geom_map[i,0] = loc[1][0]*GW
    geom_map[i,1] = loc[0][0]*GH

# generate .csv
geom_map_df = pd.DataFrame(data=geom_map)
geom_map_df.to_csv(TARGET_CSV_FPATH, index=False, header=False)
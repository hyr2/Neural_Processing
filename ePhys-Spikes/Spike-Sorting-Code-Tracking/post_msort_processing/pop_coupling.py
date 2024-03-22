# Population coupling script

import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.ndimage as sc_i

input_dir_tmp = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_3/Processed/count_analysis'

A1 = np.load(os.path.join(input_dir_tmp,'all_clus_property.npy'),allow_pickle=True)
A2 = np.load(os.path.join(input_dir_tmp,'all_clus_property_star.npy'),allow_pickle=True)

num_units_on_shank = len(A1)
num_sessions = len(A2)

# for iter_l in range(num_units_on_shank):
#     thisUnit_fr = A1[iter_l]['FR_session']
#     thisUnit_Spkcount = A1[iter_l]['spike_count_session']
#     for iter_s in range(num_sessions):
#         local_fr = thisUnit_fr[iter_s]
#         time_bins = A2[iter_s]

for iter_s in range(num_sessions):
    time_bins = A2[iter_s]
    local_fr = np.zeros(time_bins.shape,dtype = np.float64)
    for iter_l in range(num_units_on_shank):
        local_fr += A1[iter_l]['FR_session'][iter_s]    # sum with 1 ms resolution                                
    filtered_signal = sc_i.gaussian_filter1d(local_fr,10.19) # 12 ms half-width gaussian kernel
    
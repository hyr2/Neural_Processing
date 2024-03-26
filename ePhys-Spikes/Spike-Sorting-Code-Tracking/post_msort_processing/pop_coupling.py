# Population coupling script

import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.ndimage as sc_i
import scipy.signal as sc_s

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

arr_pop_rate = []    # for all sessions     (sampling rate: 1000/decimate_f)
arr_temporal = []    # time bins for all sessions
arr_mean_pop_rate = [] # mean pop rate for all sessions

decimate_f = 5

# Computing population rate for this shank (all sessions)
for iter_s in range(num_sessions):   # loop over sessions
    time_bins = A2[iter_s]
    local_fr = np.zeros(time_bins.shape,dtype = np.float64)
    for iter_l in range(num_units_on_shank):     # sum over units
        local_fr += A1[iter_l]['FR_session'][iter_s]    # sum with 1 ms resolution                                
    filtered_signal = sc_i.gaussian_filter1d(local_fr,10.19) # 12 ms half-width gaussian kernel
    # plt.plot(time_bins,filtered_signal)
    indx_l = np.arange(0,filtered_signal.shape[0],decimate_f)
    time_bins = time_bins[indx_l]
    filtered_signal = sc_s.decimate(filtered_signal,5,ftype = 'iir',zero_phase=True)
    
    # saving all population rates here
    arr_temporal.append(time_bins)
    arr_pop_rate.append(filtered_signal)
    
    # computing mean population rates
    arr_mean_pop_rate.append(np.mean(filtered_signal))
    
    
    # plt.plot(time_bins,filtered_signal)

# Population Coupling Coefficients with the population 
# (single session) iter_s
c_i = []    # population coupling of unit i
spk_c_i = []   # mean firing rate of this unit
for iter_i in range(num_units_on_shank):
    
    f_i = sc_i.gaussian_filter1d(A1[iter_i]['FR_session'][iter_s],10.19/np.sqrt(2))
    f_i_mod = A1[iter_i]['spike_count_session'][iter_s]
    
    f_j_sum = np.zeros(f_i.shape,dtype = np.float64)    #initialize array
    for iter_j in range(num_units_on_shank):        
        
        if iter_j == iter_i:
            continue
        
        filtered_signal = sc_i.gaussian_filter1d(A1[iter_j]['FR_session'][iter_s],10.19/np.sqrt(2))
        avg_fr_local = A1[iter_j]['spike_count_session'][iter_s] / A1[iter_j]['length_session'][iter_s]
        
        f_j = filtered_signal - avg_fr_local
        
        f_j_sum += f_j
    
    c_i.append(np.dot(f_i,f_j_sum) / f_i_mod)
    spk_c_i.append(f_i_mod)
    
## Raster Marginals Model (random shuffling) 
# (single session) iter_s
f_i_M = np.empty((num_units_on_shank,A1[0]['FR_session'][iter_s].shape[0]),dtype = np.int8)
for iter_i in range(num_units_on_shank):
    f_i_M[iter_i,:] = np.reshape(A1[iter_i]['FR_session'][iter_s] * 1e-3,(1,-1))

f_i_M[f_i_M > 1] = 1    # Correcting for any double spikes in the time bin. This should not happen and hence is set to 1. Thus increasing the purity of the single units
# important invariant parameters
tmp_sum_temporal = np.sum(f_i_M,dtype=np.int32,axis = 0)
custom_bins = np.linspace(0,6,7)
hist, bin_edges = np.histogram(tmp_sum_temporal, bins=custom_bins)
bin_edges = bin_edges[:-1]
plt.plot(bin_edges,hist/hist.sum(),linewidth = 3)     # prob vs #synchronous spikes

# for finding submatrix
f_i_M[f_i_M == 0] = -1
patter_M = np.array([[1,-1],[-1,1]],dtype=np.int8)
max_peak = np.prod(patter_M.shape)
# c will contain max_peak where the overlap is perfect
c = sc_s.correlate(f_i_M, patter_M, 'valid')
c = np.around(c)
c = c.astype(np.int8)
overlaps = np.where(c == max_peak)

# for swapping submatrix
rows_o = np.unique(overlaps[0])
for iter_i in rows_o:
    indx_flip_r = np.where(overlaps[0] == iter_i)[0]
    indx_flip_c = overlaps[1][indx_flip_r]
    indx_flip_r = overlaps[0][indx_flip_r]
    f_i_M[iter_i,indx_flip_c]



    
    
    
    

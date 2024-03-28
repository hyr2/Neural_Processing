# Population coupling script

import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.ndimage as sc_i
import scipy.signal as sc_s
import copy, time
import pandas as pd

def func_invariant_params(f_i_M):   
    # function computes the 2N invariant parameters after shuffling. N is the number of neurons (row index)
    tmp_sum_temporal = np.sum(f_i_M,dtype=np.int32,axis = 0)
    custom_bins = np.linspace(0,6,7)
    hist, bin_edges = np.histogram(tmp_sum_temporal, bins=custom_bins)
    bin_edges = bin_edges[:-1]

    return (bin_edges,hist,np.sum(f_i_M,axis=1))

def replace_submatrix(mat, ind1, ind2, mat_replace):
    # mat is the input matrix
    # ind1 is the rows
    # ind2 is the columns
    # mat_replace is the replacement matrix of size ind1 x ind2 sizes
  for i, index in enumerate(ind1):
    mat[index, ind2] = mat_replace[i, :]
  return mat
    

script_name = '_pop_coupling'
t = time.localtime()
current_time = time.strftime("%m_%d_%Y_%H_%M", t)
current_time = current_time + script_name
output_folder = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


input_dir_tmp = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_3/Processed/count_analysis'
input_dir_tmp1 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_1/Processed/count_analysis'

A1 = np.load(os.path.join(input_dir_tmp,'all_clus_property.npy'),allow_pickle=True)
# A3 = np.load(os.path.join(input_dir_tmp1,'all_clus_property.npy'),allow_pickle=True)
# A1 = np.concatenate((A1,A3),axis = 0)
# del A3
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
    filtered_signal = sc_s.decimate(filtered_signal,decimate_f,ftype = 'iir',zero_phase=True)
    
    # saving all population rates here
    arr_temporal.append(time_bins)
    arr_pop_rate.append(filtered_signal)
    
    # fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
    # axes.plot()
    
    # computing mean population rates
    arr_mean_pop_rate.append(np.mean(filtered_signal))
    
# Plotting for all sessions
fig, ax = plt.subplots(2,1, figsize=(10,12), dpi=100)
ax = ax.flatten()
ax[0].plot(arr_temporal[0],arr_pop_rate[0])
ax[0].set_xlim([150,151.5])
ax[0].set_ylim([0,500])
# fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
# ax = ax.flatten()
ax[1].plot(arr_temporal[1],arr_pop_rate[1])
ax[1].set_xlim([2000,2001.5])
ax[1].set_ylim([0,500])
filename_save = os.path.join(output_folder,'popRate_bsl.png')
fig.savefig(filename_save,dpi = 300)

# Plotting for all sessions
fig, ax = plt.subplots(3,1, figsize=(10,12), dpi=100)
ax = ax.flatten()
ax[0].plot(arr_temporal[2],arr_pop_rate[2])
ax[0].set_xlim([3400,3401.5])
ax[0].set_ylim([0,700])
# fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
# ax = ax.flatten()
ax[1].plot(arr_temporal[3],arr_pop_rate[3])
ax[1].set_xlim([4800,4801.5])
ax[1].set_ylim([0,700])
ax[2].plot(arr_temporal[4],arr_pop_rate[4])
ax[2].set_xlim([6000,6001.5])
ax[2].set_ylim([0,700])
filename_save = os.path.join(output_folder,'popRate_day2day7day14.png')
fig.savefig(filename_save,dpi = 300)


# Population Coupling Coefficients with the population 
# (single session) iter_s
session_ids = ['session' + str(iter_l) for iter_l in range(num_sessions)]
c_i = []    # population coupling of unit i
df_c_i = pd.DataFrame(data=None, index=range(num_units_on_shank) , columns = [session_ids])
spk_c_i = []   # mean firing rate of this unit
for iter_s in range(num_sessions):
    for iter_i in range(num_units_on_shank):
        
        f_i = sc_i.gaussian_filter1d(A1[iter_i]['FR_session'][iter_s],10.19/np.sqrt(2))
        f_i_mod = A1[iter_i]['spike_count_session'][iter_s]
        
        f_j_sum = np.zeros(f_i.shape,dtype = np.float64)    #initialize array
        for iter_j in range(num_units_on_shank):            # this is the summation over the units except for i
            if iter_j == iter_i:
                continue
            filtered_signal = sc_i.gaussian_filter1d(A1[iter_j]['FR_session'][iter_s],10.19/np.sqrt(2))
            avg_fr_local = A1[iter_j]['spike_count_session'][iter_s] / A1[iter_j]['length_session'][iter_s]
            f_j = filtered_signal - avg_fr_local
            f_j_sum += f_j
        
        c_i_local = np.dot(f_i,f_j_sum) / f_i_mod
        df_c_i.iat[iter_i,iter_s] = c_i_local
        c_i.append(c_i_local)
        spk_c_i.append(f_i_mod)
    
# plotting c_i
x = np.arange(0,num_units_on_shank)
fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
plt.bar(x,height = df_c_i.iloc[:,0],alpha = 0.5)    # baseline 1
plt.bar(x,height = df_c_i.iloc[:,1],alpha = 0.5)    # baseline 2
ax.axis('off')
filename_save = os.path.join(output_folder,'c_i_baselines.png')
fig.savefig(filename_save,dpi = 300)

fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
plt.bar(x,height = df_c_i.iloc[:,0],alpha = 0.5)    # baseline 1
plt.bar(x,height = df_c_i.iloc[:,2],alpha = 0.5)    # baseline 2
ax.axis('off')
filename_save = os.path.join(output_folder,'c_i_baselinesvsday2.png')
fig.savefig(filename_save,dpi = 300)

fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
plt.bar(x,height = df_c_i.iloc[:,0],alpha = 0.5)    # baseline 1
plt.bar(x,height = df_c_i.iloc[:,3],alpha = 0.5)    # baseline 2
ax.axis('off')
filename_save = os.path.join(output_folder,'c_i_baselinevsday7.png')
fig.savefig(filename_save,dpi = 300)

fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
plt.bar(x,height = df_c_i.iloc[:,0],alpha = 0.5)    # baseline 1
plt.bar(x,height = df_c_i.iloc[:,6],alpha = 0.5)    # baseline 2
ax.axis('off')
filename_save = os.path.join(output_folder,'c_i_baselinevsday28.png')
fig.savefig(filename_save,dpi = 300)
    
## Raster Marginals Model (random shuffling) 
# (single session) iter_s
f_i_M = np.empty((num_units_on_shank,A1[0]['FR_session'][iter_s].shape[0]),dtype = np.int8)
for iter_i in range(num_units_on_shank):
    f_i_M[iter_i,:] = np.reshape(A1[iter_i]['FR_session'][iter_s] * 1e-3,(1,-1))

f_i_M[f_i_M > 1] = 1    # Correcting for any double spikes in the time bin. This should not happen and hence is set to 1. Thus increasing the purity of the single units
# important invariant parameters
bin_edges,hist,spk_c_i = func_invariant_params(f_i_M)
fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
ax.plot(bin_edges,hist/hist.sum(),linewidth = 3)     # prob vs #synchronous spikes
filename_save = os.path.join(output_folder,'prob_synchspikes_original.png')
fig.savefig(filename_save,dpi = 300)
# for finding submatrix
f_i_M[f_i_M == 0] = -1
patter_M = np.array([[1,-1],[-1,1]],dtype=np.int8)
max_peak = np.prod(patter_M.shape)
# c will contain max_peak where the overlap is perfect
c = sc_s.correlate(f_i_M, patter_M, 'valid')
c = np.around(c)
c = c.astype(np.int8)
overlaps = np.where(c == max_peak)
f_i_M[f_i_M == -1] = 0
# for swapping submatrix
rows_o = np.unique(overlaps[0])
for iter_i in rows_o:
    indx_flip_r = np.where(overlaps[0] == iter_i)[0]
    indx_flip_c = overlaps[1][indx_flip_r]
    indx_flip_r = overlaps[0][indx_flip_r]
    
    tmp_replacement_0 = np.zeros((2,indx_flip_c.shape[0]))
    tmp_replacement_0[1,:] = 1 
    tmp_replacement_1 = copy.deepcopy(tmp_replacement_0)
    tmp_replacement_1[[0,1]] = tmp_replacement_1[[1,0]]
    
    indx_rows = [iter_i,iter_i+1]
    indx_col = list(indx_flip_c)
    f_i_M = replace_submatrix(f_i_M,indx_rows,indx_col,tmp_replacement_0)
    indx_col = list(indx_flip_c+1)
    f_i_M = replace_submatrix(f_i_M,indx_rows,indx_col,tmp_replacement_1)
    
    # f_i_M[[iter_i,iter_i+1]][:,list(indx_flip_c)] = tmp_replacement_0          # advanced non-contiguous slicing (replacement doesn't work)
    # f_i_M[[iter_i,iter_i+1]][:,list(indx_flip_c+1)] = tmp_replacement_1        # advanced non-contiguous slicing (replacement doesn't work)
    
# f_i_M is now the shuffled activity matrix
bin_edges_new,hist_new,spk_c_i_new = func_invariant_params(f_i_M)
fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
ax.plot(bin_edges_new,hist_new/hist_new.sum(),linewidth = 3)     # prob vs #synchronous spikes
filename_save = os.path.join(output_folder,'prob_synchspikes_shuffle.png')
fig.savefig(filename_save,dpi = 300)

    
    

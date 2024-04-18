#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:47:02 2024

@author: hyr2-office
"""

# Population coupling script

import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.ndimage as sc_i
import scipy.signal as sc_s
import scipy.stats as sc_ss
import copy, time
import pandas as pd
import pickle
import seaborn as sns
from natsort import natsorted

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
    

script_name = '_pop_coupling_representative'
t = time.localtime()
current_time = time.strftime("%m_%d_%Y_%H_%M", t)
current_time = current_time + script_name
output_folder = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Fig 6C: C_i of one shank over all sessions 
input_dir_tmp = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_3/Processed/count_analysis'
A1 = np.load(os.path.join(input_dir_tmp,'all_clus_property.npy'),allow_pickle=True)
A2 = np.load(os.path.join(input_dir_tmp,'all_clus_property_star.npy'),allow_pickle=True)

num_units_on_shank = len(A1)
num_sessions = len(A2)

c_mouse_rh11 = np.load(os.path.join('/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/','all_rep_depth.npy'),allow_pickle=True)
depth_list = [A1[iter_l]['depth'] for iter_l in range(num_units_on_shank)]
depth_list = np.array(depth_list,dtype = np.int16)
depth_list= depth_list

# Population Coupling Coefficients with the population 
dict_config = {}
dict_config['rh11'] = np.array([-3,-2,14,21,28,35,42,49])
df_c_i_pr = pd.DataFrame(data=None, index=range(num_units_on_shank) , columns = dict_config['rh11'].tolist())    # population coupling of unit i
for iter_s in range(num_sessions):
    for iter_i in range(num_units_on_shank):
        
        f_i = sc_i.gaussian_filter1d(A1[iter_i]['FR_session'][iter_s],10.19/np.sqrt(2))
        f_i_mod = A1[iter_i]['spike_count_session'][iter_s]
        
        f_j_sum = np.zeros(f_i.shape,dtype = np.float64)    #initialize array
        for iter_j in range(num_units_on_shank):            # this is the summation over the units except for i
            if iter_j == iter_i:
                continue
            filtered_signal = sc_i.gaussian_filter1d(A1[iter_j]['FR_session'][iter_s],10.19/np.sqrt(2))
            f_j = filtered_signal
            f_j_sum = f_j_sum + f_j
        f_j_sum = f_j_sum/(num_units_on_shank-1)    
        df_c_i_pr.iat[iter_i,iter_s] = sc_ss.pearsonr(f_j_sum,f_i)[0]        # definition used is: DOI: https://doi.org/10.7554/eLife.56053
# plotting figure 6C
df_c_i_pr['x_jitter'] = np.random.random_integers(low = -4,high = 4,size=(len(df_c_i_pr),))
for iter_ll in dict_config['rh11']:
    df_c_i_pr['depth'] = 800 - depth_list
    df_c_i_pr_bsl = df_c_i_pr.filter(items = [iter_ll,'depth','x_jitter'])
    df_c_i_pr_bsl.columns = ['Ci','depth','jitter']
    df_c_i_pr_bsl['depth'] = df_c_i_pr_bsl['depth'].astype(np.int16)
    
    
    # cmap_1 = sns.color_palette("Spectral", as_cmap=True).reversed()
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.scatterplot(data=df_c_i_pr_bsl,x = 'jitter',y='depth',ax = axes,size = 175*df_c_i_pr_bsl['Ci'],c = '#4b4b4b',edgecolor = 'k',linewidth=2,alpha = 0.9)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    axes.set_xlim(-5.5,5.5)
    filename_save = os.path.join(output_folder,f'Fig6C_day_{iter_ll}.png')
    fig.savefig(filename_save , dpi = 300,transparent=True)
    plt.close(fig)
    # axes.set_ylim(5.5,5.5)


# Population Rate representative of one shank. Fig 6A   # Computing population rate for this shank
decimate_f = 5
session_id = 21
iter_s = np.where(dict_config['rh11'] == session_id)[0][0]
df_sorted_single_session = df_c_i_pr.loc[:,session_id].sort_values( )
indx_sort = df_sorted_single_session.index
time_bins = A2[iter_s]
local_fr = np.zeros(time_bins.shape,dtype = np.float64)
for iter_l in range(num_units_on_shank):     # sum over units
    local_fr += A1[iter_l]['FR_session'][iter_s]    # sum with 1 ms resolution                                
filtered_signal = sc_i.gaussian_filter1d(local_fr,10.19) # 12 ms half-width gaussian kernel
# plt.plot(time_bins,filtered_signal)
indx_l = np.arange(0,filtered_signal.shape[0],decimate_f)
time_bins = time_bins[indx_l]
filtered_signal = sc_s.decimate(filtered_signal,decimate_f,ftype = 'iir',zero_phase=True)
fig, axes = plt.subplots(1,1, figsize=(10,2), dpi=100)
# t1 = np.where(time_bins > 4867.5)[0][0]
# t2 = np.where(time_bins > 4873)[0][0]
t1 = 0
t2 = 1000
axes.plot(time_bins[t1:t2],filtered_signal[t1:t2],c = 'k',linewidth = 2.5)
axes.set_axis_off()
filename_save = os.path.join(output_folder,'Pop_rate.png')
fig.savefig(filename_save,dpi = 300)

# Figure 6B Raster plots of representative units
raster_folder = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_3/Processed/spike_times'
files_list = natsorted(os.listdir(raster_folder))
raster_stack = []
Fs = 30e3   #Rh11 sampling rate is 30KHz
for local_file in files_list:
    with open(os.path.join('/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_3/Processed/spike_times',local_file), 'rb') as f:    # load all spike times 
        clus_local = pickle.load(f)
    clus_local = clus_local[session_id]  
    raster_stack.append(clus_local)

raster_stack = [raster_stack[iter_o] for iter_o in indx_sort.tolist()]  # sort based on coupling coefficient (see dataframe above)
fig, axes = plt.subplots(1,1, figsize=(10,6), dpi=100)  
for iter_l in range(num_units_on_shank):
    local_raster = raster_stack[iter_l]
    indx1 = np.where(local_raster/Fs > 4188.407466666667)[0][0]
    indx2 = np.where(local_raster/Fs > 4193.407466666667)[0][0]
    local_raster = local_raster[indx1:indx2]
    length_vec = local_raster.shape[0]
    y_height = [num_units_on_shank - iter_l] * length_vec
    axes.plot(local_raster,y_height,marker = "|",markersize = 15,c = 'k',alpha = 0.9,linestyle='None',markeredgewidth = 2)
axes.set_axis_off()
filename_save = os.path.join(output_folder,'Raster_RH11_shank3.png')
fig.savefig(filename_save,dpi = 300, transparent = True)
# axes.set_xlim()
# for iter_l in range(num_units_on_shank):
#     thisUnit_fr = A1[iter_l]['FR_session']
#     thisUnit_Spkcount = A1[iter_l]['spike_count_session']
#     for iter_s in range(num_sessions):
#         local_fr = thisUnit_fr[iter_s]
#         time_bins = A2[iter_s]


    
# # Plotting for all sessions
# fig, ax = plt.subplots(2,1, figsize=(10,12), dpi=100)
# ax = ax.flatten()
# ax[0].plot(arr_temporal[0],arr_pop_rate[0])
# ax[0].set_xlim([150,151.5])
# ax[0].set_ylim([0,500])
# # fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
# # ax = ax.flatten()
# ax[1].plot(arr_temporal[1],arr_pop_rate[1])
# ax[1].set_xlim([2000,2001.5])
# ax[1].set_ylim([0,500])
# filename_save = os.path.join(output_folder,'popRate_bsl.png')
# fig.savefig(filename_save,dpi = 300)

# # Plotting for all sessions
# fig, ax = plt.subplots(3,1, figsize=(10,12), dpi=100)
# ax = ax.flatten()
# ax[0].plot(arr_temporal[2],arr_pop_rate[2])
# ax[0].set_xlim([3400,3401.5])
# ax[0].set_ylim([0,700])
# # fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
# # ax = ax.flatten()
# ax[1].plot(arr_temporal[3],arr_pop_rate[3])
# ax[1].set_xlim([4800,4801.5])
# ax[1].set_ylim([0,700])
# ax[2].plot(arr_temporal[4],arr_pop_rate[4])
# ax[2].set_xlim([6000,6001.5])
# ax[2].set_ylim([0,700])
# filename_save = os.path.join(output_folder,'popRate_day2day7day14.png')
# fig.savefig(filename_save,dpi = 300)


# Population Coupling Coefficients with the population 
# (single session) iter_s

    
# # plotting c_i
# x = np.arange(0,num_units_on_shank)
# fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
# plt.bar(x,height = df_c_i.iloc[:,0],alpha = 0.5)    # baseline 1
# plt.bar(x,height = df_c_i.iloc[:,1],alpha = 0.5)    # baseline 2
# ax.axis('off')
# filename_save = os.path.join(output_folder,'c_i_baselines.png')
# fig.savefig(filename_save,dpi = 300)

# fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
# plt.bar(x,height = df_c_i.iloc[:,0],alpha = 0.5)    # baseline 1
# plt.bar(x,height = df_c_i.iloc[:,2],alpha = 0.5)    # baseline 2
# ax.axis('off')
# filename_save = os.path.join(output_folder,'c_i_baselinesvsday14.png')
# fig.savefig(filename_save,dpi = 300)

# fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
# plt.bar(x,height = df_c_i.iloc[:,0],alpha = 0.5)    # baseline 1
# plt.bar(x,height = df_c_i.iloc[:,3],alpha = 0.5)    # baseline 2
# ax.axis('off')
# filename_save = os.path.join(output_folder,'c_i_baselinevsday21.png')
# fig.savefig(filename_save,dpi = 300)

# fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
# plt.bar(x,height = df_c_i.iloc[:,0],alpha = 0.5)    # baseline 1
# plt.bar(x,height = df_c_i.iloc[:,4],alpha = 0.5)    # baseline 2
# ax.axis('off')
# filename_save = os.path.join(output_folder,'c_i_baselinevsday28.png')
# fig.savefig(filename_save,dpi = 300)
    


'''
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
'''
    
    

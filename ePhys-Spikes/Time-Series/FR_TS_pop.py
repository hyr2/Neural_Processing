#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:15:37 2022

@author: hyr2-office
"""

import scipy.io as sio # Import function to read data.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from natsort import natsorted
import pandas as pd

# Plotting fonts
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=16)          # controls default text sizes

# Used to process one single animal, all sessions
# Make sure the codes: population_analysis.py is run as well as func_CE_BarrelCortex.m before this (see the rickshaw post processing file)

def sort_by_shank(type_local,shank_num_local):
    # This function is used to assign the excitatory and inhibitory type neurons to their respective shanks (A,B,C,D)
    # INPUTS: 
    #    type_local: array of boolean for excitatory/inhibitory neurons (1 -> excitatory_cell, 0 -> not excitatory_cell)
    #    shank_num_local: an array the same size as type_local containing the shank info of each cell/cluster 
    output_main = np.zeros([4,])
    for iter in range(4):
        output_local = np.logical_and(type_local,shank_num_local == iter+1) 
        output_main[iter] = np.sum(output_local)
        
    return output_main
    
def sort_cell_type(input_arr):
    # Function counts the number of wide, narrow and pyramidal cells from the matlab output (.mat file called pop_celltypes.mat)
    output_arr = np.zeros([3,],dtype = np.int16)
    if not input_arr.shape:
        return output_arr
    else:
        for iter in range(input_arr.shape[1]):
            str_celltype = input_arr[0][iter]
            if str_celltype == 'Pyramidal Cell':
                output_arr[0] += 1 
            elif str_celltype == 'Narrow Interneuron':
                output_arr[1] += 1 
            elif str_celltype == 'Wide Interneuron':
                output_arr[2] += 1 
        return output_arr
    
def convert2df(T2P_allsessions):
    # Function is being used to organize the array for trough to peak time (ms) 
    # the input is a dictionary of multiple numpy arrays of different lengths. 
    # Each index of the dictionary represents one session
    # The output is a dataframe to be used for better data organization and plotting
    df_excit = pd.DataFrame(
            {
                "index": [],
                "T2P": []
            }
        )
    df_inhib = pd.DataFrame(
            {
                "index": [],
                "T2P": []
            }
        )
    T2P = pd.Series([],dtype = 'float')
    for iter_local in range(len(T2P_allsessions)):
        T2P = pd.concat([T2P,pd.Series(np.squeeze(T2P_allsessions[iter_local]))])
    
    tmp_bool = np.array(T2P > 0.55)
    df_excit.T2P = T2P[tmp_bool]
    df_excit.index = np.linspace(0,df_excit.T2P.shape[0]-1,df_excit.T2P.shape[0])
    tmp_bool = np.array(T2P <= 0.55)
    df_inhib.T2P = T2P[tmp_bool]
    df_inhib.index = np.linspace(0,df_inhib.T2P.shape[0]-1,df_inhib.T2P.shape[0])
    return df_excit,df_inhib

source_dir = '/home/hyr2-office/Documents/Data/NVC/RH-8/'
rmv_bsl = input('Baselines to remove (specify as index: e.g: 0, 2)? Select -1 for no baselines.\n')             # specify what baseline datasets need to be removed from the analysis
source_dir_list = natsorted(os.listdir(source_dir))
# Preparing variables
rmv_bsl = rmv_bsl.split(',')
rmv_bsl = np.asarray(rmv_bsl, dtype = np.int8)

if not np.any(rmv_bsl == -1):
    source_dir_list = np.delete(source_dir_list,rmv_bsl)
    source_dir_list = source_dir_list.tolist()

# source_dir_list = natsorted(os.listdir(source_dir))

# x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14','Day 21','Day 28','Day 56']  # RH3 (reject baselines 0 and 2)
# linear_xaxis = np.array([-2,-1,2,7,14,21,28,56]) 
# x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14 ','Day 21','Day 28','Day 42'] # BC7 (reject baseline 0)
# linear_xaxis = np.array([-2,-1,2,7,14,21,28,42]) 
# x_ticks_labels = ['bl-1','bl-2','Day 2','Day 9','Day 14 ','Day 21','Day 28','Day 35','Day 49'] # BC6 (stroke not formed at all. Data should be rejected)
# linear_xaxis = np.array([-2,-1,2,9,14,21,28,35,49]) 
# x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14','Day 21','Day 47'] # B-BC5
# linear_xaxis = np.array([-2,-1,2,7,14,21,47]) 
# x_ticks_labels = ['bl-1','bl-2','bl-3','Day 2','Day 7','Day 14 ','Day 24','Day 28','Day 35','Day 42','Day 49','Day 56'] # R-H7 (main)
# linear_xaxis = np.array([-3,-2,-1,2,7,14,24,28,35,42,49,56]) # 24 special for rh7
# x_ticks_labels = ['bl-1','Day 2','Day 7','Day 14 ','Day 21','Day 42'] # BC8 
# linear_xaxis = np.array([-3,-2,-1,2,2,7,8,14,21,54])            
x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14 ','Day 21','Day 42'] # R-H8 
linear_xaxis = np.array([-2,-1,2,7,14,21,28,35,42,49])            

x_ticks_labels = linear_xaxis

pop_stats = {}
pop_stats_cell = {}
names_datasets = {}
iter = 0
# Loading all longitudinal data into dictionaries 
for name in source_dir_list:
    if os.path.isdir(os.path.join(source_dir,name)):
        folder_loc_mat = os.path.join(source_dir,name)
        if os.path.isdir(folder_loc_mat):
            pop_stats[iter] = sio.loadmat(os.path.join(folder_loc_mat,'Processed/count_analysis/population_stat_responsive_only.mat'))
            pop_stats_cell[iter] = sio.loadmat(os.path.join(folder_loc_mat,'Processed/cell_type/pop_celltypes.mat'))
            names_datasets[iter] = name
            iter += 1
        
act_nclus = np.zeros([len(pop_stats),4])
act_FR = np.zeros([len(pop_stats),4])
suppressed_nclus = np.zeros([len(pop_stats),4])
inh_FR = np.zeros([len(pop_stats),4])
nor_nclus = np.zeros([len(pop_stats),4])
N_chan_shank = np.zeros([len(pop_stats),4])
act_nclus_total = np.zeros([len(pop_stats),])
suppressed_nclus_total = np.zeros([len(pop_stats),])
celltype_total = np.zeros([len(pop_stats),3])
excitatory_cell = np.zeros([len(pop_stats),4]) # by shank
inhibitory_cell = np.zeros([len(pop_stats),4]) # by shank
# celltype_excit = np.zeros([len(pop_stats),3])
# celltype_inhib = np.zeros([len(pop_stats),3])
T2P_allsessions = {}    # dictionary of 1D numpy arrays
for iter in range(len(pop_stats)):
    # population extraction from dictionaries
    
    # FR goes up (activated neurons)
    act_nclus[iter,0] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[0]     # Shank A
    act_nclus[iter,1] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[1]     # Shank B 
    act_nclus[iter,2] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[2]     # Shank C
    act_nclus[iter,3] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[3]     # Shank D
    act_FR[iter,0] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[0]
    act_FR[iter,1] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[1]
    act_FR[iter,2] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[2]
    act_FR[iter,3] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[3]
    # FR goes down (suppressed activity)
    suppressed_nclus[iter,0] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[0]     # Shank A
    suppressed_nclus[iter,1] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[1]     # Shank B 
    suppressed_nclus[iter,2] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[2]     # Shank C
    suppressed_nclus[iter,3] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[3]     # Shank C
    inh_FR[iter,0] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[0]
    inh_FR[iter,1] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[1]
    inh_FR[iter,2] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[2]
    inh_FR[iter,3] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[3]
    # No response clusters
    nor_nclus[iter,0] = np.squeeze(pop_stats[iter]['nor_nclus_by_shank'])[0] 
    nor_nclus[iter,1] = np.squeeze(pop_stats[iter]['nor_nclus_by_shank'])[1] 
    nor_nclus[iter,2] = np.squeeze(pop_stats[iter]['nor_nclus_by_shank'])[2] 
    nor_nclus[iter,3] = np.squeeze(pop_stats[iter]['nor_nclus_by_shank'])[3] 
    # Number of channels per shank in each session
    N_chan_shank[iter,0] = np.squeeze(pop_stats[iter]['numChan_perShank'])[0]
    N_chan_shank[iter,1] = np.squeeze(pop_stats[iter]['numChan_perShank'])[1]
    N_chan_shank[iter,2] = np.squeeze(pop_stats[iter]['numChan_perShank'])[2]
    N_chan_shank[iter,3] = np.squeeze(pop_stats[iter]['numChan_perShank'])[3]
    
    act_nclus_total[iter] = np.sum(act_nclus[iter,:])
    suppressed_nclus_total[iter] = np.sum(suppressed_nclus[iter,:])
    
    # cell type extraction from dictionaries
    tmp = pop_stats_cell[iter]['celltype']
    celltype_total[iter,:] = sort_cell_type(tmp)
    
    # excitatory and inhibitory neuron populations
    excitatory_cell[iter,:] = sort_by_shank(pop_stats_cell[iter]['type_excit'],pop_stats_cell[iter]['shank_num'])
    inhibitory_cell[iter,:] = sort_by_shank(pop_stats_cell[iter]['type_inhib'],pop_stats_cell[iter]['shank_num'])
    
    # Saving T2P for global histogram
    T2P_allsessions[iter] = pop_stats_cell[iter]['troughToPeak']
    

# total neurons by cell type
# celltype_total = celltype_excit + celltype_inhib
total_activity_act = act_FR * act_nclus
work_amount_act = act_FR / act_nclus
# Saving mouse summary for averaging 
full_mouse_ephys = {}
full_mouse_ephys['List'] = names_datasets
full_mouse_ephys['act_nclus_total'] = act_nclus_total
full_mouse_ephys['suppressed_nclus_total'] = suppressed_nclus_total
full_mouse_ephys['nor_nclus_total'] = nor_nclus
full_mouse_ephys['N_chan_shank'] = N_chan_shank
full_mouse_ephys['excitatory_cell'] = excitatory_cell
full_mouse_ephys['inhibitory_cell'] = inhibitory_cell
full_mouse_ephys['act_nclus'] = act_nclus
full_mouse_ephys['suppressed_nclus'] = suppressed_nclus
full_mouse_ephys['x_ticks_labels'] = x_ticks_labels
full_mouse_ephys['celltype_total'] = celltype_total
full_mouse_ephys['T2P'] = T2P_allsessions
full_mouse_ephys['total_activity_act'] = total_activity_act
full_mouse_ephys['FR_act'] = act_FR
sio.savemat(os.path.join(source_dir,'full_mouse_ephys.mat'), full_mouse_ephys)

# Plot of cell types + activated/suppressed + excitatory/inhibitory neurons
filename_save = os.path.join(source_dir,'Population_analysis_cell_activation.png')
f, a = plt.subplots(2,3)
a[0,0].set_ylabel('Pop. Count')
a[1,0].set_ylabel('Pop. Count')
a[1,0].set_xlabel('Days')
a[1,1].set_xlabel('Days')
a[1,2].set_xlabel('Days')
len_str = 'Population Analysis'
f.suptitle(len_str)
a[0,0].set_title("Activated Neurons")
a[0,0].plot(x_ticks_labels,act_nclus[:,0],'r', lw=1.5)
a[0,0].plot(x_ticks_labels,act_nclus[:,1],'g', lw=1.5)
a[0,0].plot(x_ticks_labels,act_nclus[:,2],'b', lw=1.5)
a[0,0].plot(x_ticks_labels,act_nclus[:,3],'y', lw=1.5)
a[0,0].legend(['ShankA', 'ShankB','ShankC','ShankD'])
a[1,0].set_title("Suppressed Neurons")
a[1,0].plot(x_ticks_labels,suppressed_nclus[:,0],'r', lw=1.5)
a[1,0].plot(x_ticks_labels,suppressed_nclus[:,1],'g', lw=1.5)
a[1,0].plot(x_ticks_labels,suppressed_nclus[:,2],'b', lw=1.5)
a[1,0].plot(x_ticks_labels,suppressed_nclus[:,3],'y', lw=1.5)
a[1,0].legend(['ShankA','ShankB','ShankC', 'ShankD'])
a[1,1].set_title("All neurons")
a[1,1].plot(x_ticks_labels,act_nclus_total,'r', lw=1.5)
a[1,1].plot(x_ticks_labels,suppressed_nclus_total,'b', lw=1.5)
a[1,1].legend(['Activated','Suppressed'])
a[0,2].set_title("Excitatory Neurons (Waveform Analysis)")
a[0,2].plot(x_ticks_labels,excitatory_cell[:,0],'r', lw=1.5)
a[0,2].plot(x_ticks_labels,excitatory_cell[:,1],'g', lw=1.5)
a[0,2].plot(x_ticks_labels,excitatory_cell[:,2],'b', lw=1.5)
a[0,2].plot(x_ticks_labels,excitatory_cell[:,3],'y', lw=1.5)
a[0,2].legend(['ShankA', 'ShankB','ShankC','ShankD'])
a[1,2].set_title("Inhibitory Neurons (Waveform Analysis)")
a[1,2].plot(x_ticks_labels,inhibitory_cell[:,0],'r', lw=1.5)
a[1,2].plot(x_ticks_labels,inhibitory_cell[:,1],'g', lw=1.5)
a[1,2].plot(x_ticks_labels,inhibitory_cell[:,2],'b', lw=1.5)
a[1,2].plot(x_ticks_labels,inhibitory_cell[:,3],'y', lw=1.5)
a[1,2].legend(['ShankA', 'ShankB','ShankC','ShankD'])
a[0,1].set_title("Cell Types")
a[0,1].plot(x_ticks_labels,celltype_total[:,0],'g', lw=1.5)
a[0,1].plot(x_ticks_labels,celltype_total[:,1],'k', lw=1.5)
a[0,1].plot(x_ticks_labels,celltype_total[:,2],'y', lw=1.5)
a[0,1].legend(['Pyramidal','Narrow','Wide'])
f.set_size_inches((20, 8), forward=False)
plt.savefig(filename_save,format = 'png')
plt.close(f)

# Plot for Trough2Peak latency (in ms)
tmp_str = source_dir.split('/')[-2:]
tmp_str = ' '.join(tmp_str)
if tmp_str[-1] == ' ':
    tmp_str = tmp_str[:-1]
filename_save = os.path.join(source_dir,'TP_latency_histogram_' + tmp_str + '.png')
df_excit,df_inhib = convert2df(T2P_allsessions)
ax1 = sns.histplot(data=df_excit, x="T2P", color="red", label="Trough to Peak", kde=True)
sns.histplot(data=df_inhib, x="T2P", color="skyblue", label="Trough to Peak", kde=True, ax = ax1)
ax1.set_xlabel('Trough to Peak (ms)')
f = plt.gcf()
f.set_size_inches((12, 6), forward=False)
plt.savefig(filename_save,format = 'png')
plt.close(f)
# Plots for average FR
filename_save = os.path.join(source_dir,'FR_analysis_cell_activation.png')
f, a = plt.subplots(1,2)
a[0].set_ylabel('Avg FR x # units')
a[1].set_ylabel('Avg FR')
a[0].set_xlabel('Days')
a[1].set_xlabel('Days')
len_str = 'FR analysis'
f.suptitle(len_str)
a[0].set_title("Avg FR by shank of activated neurons")
a[0].plot(x_ticks_labels,total_activity_act[:,0],'r', lw=1.5)
a[0].plot(x_ticks_labels,total_activity_act[:,1],'g', lw=1.5)
a[0].plot(x_ticks_labels,total_activity_act[:,2],'b', lw=1.5)
a[0].plot(x_ticks_labels,total_activity_act[:,3],'y', lw=1.5)
a[0].legend(['ShankA','ShankB','ShankC', 'ShankD'])
a[1].set_title("Avg FR by shank of activated neurons")
a[1].plot(x_ticks_labels,work_amount_act[:,0],'r', lw=1.5)
a[1].plot(x_ticks_labels,work_amount_act[:,1],'g', lw=1.5)
a[1].plot(x_ticks_labels,work_amount_act[:,2],'b', lw=1.5)
a[1].plot(x_ticks_labels,work_amount_act[:,3],'y', lw=1.5)
a[1].legend(['ShankA','ShankB','ShankC', 'ShankD'])
f.set_size_inches((12, 6), forward=False)
plt.savefig(filename_save,format = 'png')
plt.close(f)


# plt.figure()
# plt.bar(x_ticks_labels, act_nclus[:,0])
# plt.figure()
# plt.bar(x_ticks_labels, act_nclus[:,1])
# plt.figure()
# plt.bar(x_ticks_labels, act_nclus[:,2])
# plt.figure()
# plt.bar(x_ticks_labels, suppressed_nclus[:,0])
# plt.figure()
# plt.bar(x_ticks_labels, suppressed_nclus[:,1])
# plt.figure()
# plt.bar(x_ticks_labels, suppressed_nclus[:,2])




# Extract templates and analyze their cell types
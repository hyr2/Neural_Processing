#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:15:37 2022

@author: hyr2-office
"""

import scipy.io as sio # Import function to read data.
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import pandas as pd

# Used to process one single animal, all sessions
# Make sure the codes: population_analysis.py is run as well as hanlin_buz_ce.m

def sort_cell_type(input_arr):
    # Function counts the number of wide, narrow and pyramidal cells from the matlab output
    output_arr = np.zeros([3,],dtype = np.int16)
    if not input_arr.shape:
        return output_arr
    else:
        for iter in range(input_arr.shape[0]):
            str_celltype = input_arr[iter][0]
            if str_celltype == 'Pyramidal Cell':
                output_arr[0] += 1 
            elif str_celltype == 'Narrow Interneuron':
                output_arr[1] += 1 
            elif str_celltype == 'Wide Interneuron':
                output_arr[2] += 1 
        return output_arr
    

source_dir = '/home/hyr2-office/Documents/Data/NVC/BC8/'
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
x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14','Day 21','Day 47'] # B-BC5
linear_xaxis = np.array([-2,-1,2,7,14,21,47]) 
# x_ticks_labels = ['bl-1','bl-2','bl-3','Day 2','Day 7','Day 14 ','Day 24','Day 28','Day 35','Day 42','Day 49','Day 56'] # R-H7 (main)
# linear_xaxis = np.array([-3,-2,-1,2,7,14,24,28,35,42,49,56]) # 24 special for rh7
# x_ticks_labels = ['bl-1','Day 2','Day 7','Day 14 ','Day 21','Day 42'] # BC8 
# linear_xaxis = np.array([-3,-2,-1,2,2,7,7,8,14,21,54]) 

x_ticks_labels = linear_xaxis

pop_stats = {}
pop_stats_cell = {}
iter = 0
# Loading all longitudinal data into dictionaries 
for name in source_dir_list:
    if os.path.isdir(os.path.join(source_dir,name)):
        folder_loc_mat = os.path.join(source_dir,name)
        if os.path.isdir(folder_loc_mat):
            pop_stats[iter] = sio.loadmat(os.path.join(folder_loc_mat,'Processed/count_analysis/population_stat_responsive_only.mat'))
            pop_stats_cell[iter] = sio.loadmat(os.path.join(folder_loc_mat,'Processed/cell_type/pop_celltypes.mat'))
            iter += 1
        
act_nclus = np.zeros([len(pop_stats),4])
act_FR = np.zeros([len(pop_stats),4])
inhib_nclus = np.zeros([len(pop_stats),4])
inh_FR = np.zeros([len(pop_stats),4])
act_nclus_total = np.zeros([len(pop_stats),])
inhib_nclus_total = np.zeros([len(pop_stats),])
celltype_excit = np.zeros([len(pop_stats),3])
celltype_inhib = np.zeros([len(pop_stats),3])
for iter in range(len(pop_stats)):
    # population extraction from dictionaries
    act_nclus[iter,0] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[0]     # Shank A
    act_nclus[iter,1] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[1]     # Shank B 
    act_nclus[iter,2] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[2]     # Shank C
    act_nclus[iter,3] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[3]     # Shank D
    act_FR[iter,0] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[0]
    act_FR[iter,1] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[1]
    act_FR[iter,2] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[2]
    act_FR[iter,3] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[3]
    
    inhib_nclus[iter,0] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[0]     # Shank A
    inhib_nclus[iter,1] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[1]     # Shank B 
    inhib_nclus[iter,2] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[2]     # Shank C
    inhib_nclus[iter,3] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[3]     # Shank C
    inh_FR[iter,0] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[0]
    inh_FR[iter,1] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[1]
    inh_FR[iter,2] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[2]
    inh_FR[iter,3] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[3]
    
    # cell type extraction from dictionaries
    act_nclus_total[iter] = np.sum(act_nclus[iter,:])
    inhib_nclus_total[iter] = np.sum(inhib_nclus[iter,:])
    tmp = pop_stats_cell[iter]['type_excit']
    tmp = np.squeeze(tmp)
    celltype_excit[iter,:] = sort_cell_type(tmp)
    tmp = pop_stats_cell[iter]['type_inhib']
    tmp = np.squeeze(tmp)
    celltype_inhib[iter,:] = sort_cell_type(tmp)
    # celltype_excit[iter]

# total neurons by cell type
celltype_total = celltype_excit + celltype_inhib
total_activity_act = act_FR * act_nclus
work_amount_act = act_FR / act_nclus
# Saving mouse summary for averaging 
full_mouse_ephys = {}
full_mouse_ephys['act_nclus_total'] = act_nclus_total
full_mouse_ephys['inhib_nclus_total'] = inhib_nclus_total
full_mouse_ephys['celltype_excit'] = celltype_excit
full_mouse_ephys['celltype_inhib'] = celltype_inhib
full_mouse_ephys['act_nclus'] = act_nclus
full_mouse_ephys['inhib_nclus'] = inhib_nclus
full_mouse_ephys['x_ticks_labels'] = x_ticks_labels
full_mouse_ephys['celltype_total'] = celltype_total
full_mouse_ephys['total_activity_act'] = total_activity_act
full_mouse_ephys['FR_act'] = act_FR
sio.savemat(os.path.join(source_dir,'full_mouse_ephys.mat'), full_mouse_ephys)

# Add subplot and layout of figure
filename_save = os.path.join(source_dir,'Population_analysis_cell_activation.png')
f, a = plt.subplots(2,3)
a[0,0].set_ylabel('Pop. Count')
a[1,0].set_ylabel('Pop. Count')
a[1,0].set_xlabel('Days')
a[1,1].set_xlabel('Days')
a[1,2].set_xlabel('Days')
len_str = 'Population Analysis'
f.suptitle(len_str)
a[0,0].set_title("Excitatory Neurons")
a[0,0].plot(x_ticks_labels,act_nclus[:,0],'r', lw=1.5)
a[0,0].plot(x_ticks_labels,act_nclus[:,1],'g', lw=1.5)
a[0,0].plot(x_ticks_labels,act_nclus[:,2],'b', lw=1.5)
a[0,0].plot(x_ticks_labels,act_nclus[:,3],'y', lw=1.5)
a[0,0].legend(['ShankA', 'ShankB','ShankC','ShankD'])
a[1,0].set_title("Inhibitory Neurons")
a[1,0].plot(x_ticks_labels,inhib_nclus[:,0],'r', lw=1.5)
a[1,0].plot(x_ticks_labels,inhib_nclus[:,1],'g', lw=1.5)
a[1,0].plot(x_ticks_labels,inhib_nclus[:,2],'b', lw=1.5)
a[1,0].plot(x_ticks_labels,inhib_nclus[:,3],'y', lw=1.5)
a[1,0].legend(['ShankA','ShankB','ShankC', 'ShankD'])
a[1,1].set_title("All neurons")
a[1,1].plot(x_ticks_labels,act_nclus_total,'r', lw=1.5)
a[1,1].plot(x_ticks_labels,inhib_nclus_total,'b', lw=1.5)
a[1,1].legend(['Excitatory','Inhibitory'])
a[0,2].set_title("Excitatory Neurons")
a[0,2].plot(x_ticks_labels,celltype_excit[:,0],'g+', lw=1.5)
a[0,2].plot(x_ticks_labels,celltype_excit[:,1],'k+', lw=1.5)
a[0,2].plot(x_ticks_labels,celltype_excit[:,2],'y+', lw=1.5)
a[0,2].legend(['Pyramidal','Narrow','Wide'])
a[1,2].set_title("Inhibitory Neurons")
a[1,2].plot(x_ticks_labels,celltype_inhib[:,0],'g', lw=1.5)
a[1,2].plot(x_ticks_labels,celltype_inhib[:,1],'k', lw=1.5)
a[1,2].plot(x_ticks_labels,celltype_inhib[:,2],'y', lw=1.5)
a[1,2].legend(['Pyramidal','Narrow','Wide'])
a[0,1].set_title("Cell Types")
a[0,1].plot(x_ticks_labels,celltype_total[:,0],'g', lw=1.5)
a[0,1].plot(x_ticks_labels,celltype_total[:,1],'k', lw=1.5)
a[0,1].plot(x_ticks_labels,celltype_total[:,2],'y', lw=1.5)
a[0,1].legend(['Pyramidal','Narrow','Wide'])
f.set_size_inches((20, 6), forward=False)
plt.savefig(filename_save,format = 'png')
plt.close(f)

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
# plt.bar(x_ticks_labels, inhib_nclus[:,0])
# plt.figure()
# plt.bar(x_ticks_labels, inhib_nclus[:,1])
# plt.figure()
# plt.bar(x_ticks_labels, inhib_nclus[:,2])




# Extract templates and analyze their cell types
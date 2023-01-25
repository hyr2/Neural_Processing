#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:38:04 2022

@author: hyr2-office
"""
# Need to add BC8
# A poorly written code
# Used to combine animals 
# Must run FR_TS_pop.py for each animal before hand
# Performs linear interlpolation for missing/bad data points (hard coded)
# Shank buckets: 100<X<300 um, X>300um, outside, S2 [shanks too close to the PW have been rejected from this analysis]
 
import scipy.io as sio # Import function to read data.
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import pandas as pd
import seaborn as sns
from copy import deepcopy
from Support import interp_session_loss

# Plotting fonts
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=16)          # controls default text sizes

mouse_rh3 = '/home/hyr2-office/Documents/Data/NVC/RH-3/'
mouse_bc7 = '/home/hyr2-office/Documents/Data/NVC/BC7/'
mouse_bc6 = '/home/hyr2-office/Documents/Data/NVC/BC6/'
mouse_rh7 = '/home/hyr2-office/Documents/Data/NVC/RH-7/'
mouse_bbc5 = '/home/hyr2-office/Documents/Data/NVC/B-BC5/'
mouse_bc8 = '/home/hyr2-office/Documents/Data/NVC/BC8/'
mouse_rh8 = '/home/hyr2-office/Documents/Data/NVC/RH-8/'
mouse_rh9 = '/home/hyr2-office/Documents/Data/NVC/RH-9/'

output_folder = '/home/hyr2-office/Documents/Data/NVC/'

mouse_rh3 = os.path.join(mouse_rh3, 'full_mouse_ephys.mat')
mouse_bc7 = os.path.join(mouse_bc7, 'full_mouse_ephys.mat')
mouse_bc6 = os.path.join(mouse_bc6, 'full_mouse_ephys.mat')
mouse_rh7 = os.path.join(mouse_rh7, 'full_mouse_ephys.mat')
mouse_bbc5  = os.path.join(mouse_bbc5 , 'full_mouse_ephys.mat')
mouse_rh8 = os.path.join(mouse_rh8, 'full_mouse_ephys.mat')
mouse_rh9 = os.path.join(mouse_rh9, 'full_mouse_ephys.mat')
mouse_bc8 = os.path.join(mouse_bc8,'full_mouse_ephys.mat')

# Creating dictionaries
dict_rh3 = sio.loadmat(mouse_rh3)
dict_bc7 = sio.loadmat(mouse_bc7)
# dict_bc6 = sio.loadmat(mouse_bc6)
dict_rh7 = sio.loadmat(mouse_rh7)
dict_bbc5 = sio.loadmat(mouse_bbc5)
dict_bc8 = sio.loadmat(mouse_bc8)
dict_rh8 = sio.loadmat(mouse_rh8)
# dict_rh9 = sio.loadmat(mouse_rh9)

dict_dict =  {'dict_rh3':deepcopy(dict_rh3), 'dict_bc7':deepcopy(dict_bc7), 'dict_rh7':deepcopy(dict_rh7), 'dict_bbc5':deepcopy(dict_bbc5), 'dict_bc8':deepcopy(dict_bc8), 'dict_rh8':deepcopy(dict_rh8)}  # deepcopy used to ensure new memory location is assigned
dict_rh3.clear()
dict_bc7.clear()
# dict_bc6.clear()
dict_rh7.clear()
dict_bbc5.clear()
dict_bc8.clear()
dict_rh8.clear()


# pop_stats = {}
# pop_stats[0] = sio.loadmat(mouse_rh3)
# pop_stats[1] = sio.loadmat(mouse_bc7)
# pop_stats[2] = sio.loadmat(mouse_bc6)
# pop_stats[3] = sio.loadmat(mouse_rh7)

day_axis_ideal = np.array([-3,-2,-1,2,7,14,21,28,35,42,49,56])
x_axis_time = day_axis_ideal

# Four categories (Dr.Lan signs off on these)
D_less300 = {}
D_great300 = {}
outside_barrel = {}
S2 = {}

# de-activated neurons (FR goes down)
# D_less150[0] = pop_stats[0]['inhib_nclus'][:,0]           
# D_great150_less300[0] = pop_stats[0]['inhib_nclus'][:,1]
# D_great150_less300[1] = pop_stats[1]['inhib_nclus'][:,0]
# D_great300[0] = pop_stats[1]['inhib_nclus'][:,2]
# D_great300[1] = pop_stats[2]['inhib_nclus'][:,2]
# outside_barrel[0] = pop_stats[1]['inhib_nclus'][:,1] 
# outside_barrel[1] = pop_stats[0]['inhib_nclus'][:,2]
# S2[0] = pop_stats[1]['inhib_nclus'][:,3] 
# S2[1] = pop_stats[2]['inhib_nclus'][:,3] 

# # Fixing data
# D_less150[0] = np.append(D_less150[0],D_less150[0][-1])
# D_great150_less300[0] = np.append(D_great150_less300[0],D_great150_less300[0][-1])
# D_great300[1] = np.delete(D_great300[1],-2)
# D_great300[1][-2] = np.average([D_great300[1][-3],D_great300[1][-1]]) 
# outside_barrel[1] = np.append(outside_barrel[1],np.average([outside_barrel[1][-1],outside_barrel[1][-2]]))
# S2[1] = np.delete(S2[1],-2)
# S2[1][-2] = np.average([S2[1][-3],S2[1][-1]]) 

# # Add subplot and layout of figure
# x_ticks_labels = pop_stats[1]['x_ticks_labels']
# f, a = plt.subplots(1,1)
# a.set_ylabel('Pop. Count')
# len_str = 'Population Analysis'
# f.suptitle(len_str)
# a.set_title("Inhibitory Neurons")
# a.plot(x_ticks_labels,D_less150[0],'r', lw=1.5)
# temp_1 = np.mean(list(D_great150_less300.values()),axis=0)
# temp_2 = np.mean(list(D_great300.values()),axis=0)
# temp_avg = (temp_1 + temp_2)/2
# a.plot(x_ticks_labels,temp_avg,'g', lw=1.5)
# a.plot(x_ticks_labels,np.mean(list(outside_barrel.values()),axis=0),'k', lw=1.5)
# a.legend(['<150um','D > 150 um','Outside BC'])
# f.set_size_inches((10, 6), forward=False)
# plt.savefig(filename_save,format = 'png')
# plt.close(f)

# Data fixing (BC7) # see onenote for reasoning 
dict_dict['dict_bc7']['act_nclus'][-1,0] = np.rint((dict_dict['dict_bc7']['act_nclus'][-2,0] + dict_dict['dict_bc7']['act_nclus'][-3,0])/2)
# Data fixing B-BC5 # see onenote for reasoning 
dict_dict['dict_bbc5']['act_nclus'][0,3] = dict_dict['dict_bbc5']['act_nclus'][1,3]
# Data fixing RH3 # See onenote for reasoning
dict_dict['dict_rh3']['celltype_total'][-1,:] = np.rint((dict_dict['dict_rh3']['celltype_total'][-2,:] + dict_dict['dict_rh3']['celltype_total'][-3,:])/2)

# Data fixing baselines (interpolation should not occur in the baseline ie baseline shouldn't be linearly interpolated as to the day 2. Hence segments of the curves is important)
dict_dict['dict_bc7']['nor_nclus_total'] = np.insert(dict_dict['dict_bc7']['nor_nclus_total'],[0],np.mean(dict_dict['dict_bc7']['nor_nclus_total'][0:2,:],axis = 0),axis=0)
# dict_dict['dict_rh7']['act_nclus'] = np.insert(dict_dict['dict_rh7']['act_nclus'],[0],np.mean(dict_dict['dict_rh7']['act_nclus'][0:2,:],axis = 0),axis=0) # b/c rh7 has 3 good baselines
dict_dict['dict_rh3']['nor_nclus_total'] = np.insert(dict_dict['dict_rh3']['nor_nclus_total'],[0],np.mean(dict_dict['dict_rh3']['nor_nclus_total'][0:2,:],axis = 0),axis=0)
dict_dict['dict_bbc5']['nor_nclus_total'] = np.insert(dict_dict['dict_bbc5']['nor_nclus_total'],[0],np.mean(dict_dict['dict_bbc5']['nor_nclus_total'][0:2,:],axis = 0),axis=0)
dict_dict['dict_rh8']['nor_nclus_total'] = np.insert(dict_dict['dict_rh8']['nor_nclus_total'],[0],np.mean(dict_dict['dict_rh8']['nor_nclus_total'][0:2,:],axis = 0),axis=0)
dict_dict['dict_bc7']['act_nclus'] = np.insert(dict_dict['dict_bc7']['act_nclus'],[0],np.mean(dict_dict['dict_bc7']['act_nclus'][0:2,:],axis = 0),axis=0)
# dict_dict['dict_rh7']['act_nclus'] = np.insert(dict_dict['dict_rh7']['act_nclus'],[0],np.mean(dict_dict['dict_rh7']['act_nclus'][0:2,:],axis = 0),axis=0) # b/c rh7 has 3 good baselines
dict_dict['dict_rh3']['act_nclus'] = np.insert(dict_dict['dict_rh3']['act_nclus'],[0],np.mean(dict_dict['dict_rh3']['act_nclus'][0:2,:],axis = 0),axis=0)
dict_dict['dict_bbc5']['act_nclus'] = np.insert(dict_dict['dict_bbc5']['act_nclus'],[0],np.mean(dict_dict['dict_bbc5']['act_nclus'][0:2,:],axis = 0),axis=0)
dict_dict['dict_rh8']['act_nclus'] = np.insert(dict_dict['dict_rh8']['act_nclus'],[0],np.mean(dict_dict['dict_rh8']['act_nclus'][0:2,:],axis = 0),axis=0)
dict_dict['dict_bc7']['celltype_total'] = np.insert(dict_dict['dict_bc7']['celltype_total'],[0],np.mean(dict_dict['dict_bc7']['celltype_total'][0:2,:],axis = 0),axis=0)
# dict_dict['dict_rh7']['act_nclus'] = np.insert(dict_dict['dict_rh7']['act_nclus'],[0],np.mean(dict_dict['dict_rh7']['act_nclus'][0:2,:],axis = 0),axis=0) # b/c rh7 has 3 good baselines
dict_dict['dict_rh3']['celltype_total'] = np.insert(dict_dict['dict_rh3']['celltype_total'],[0],np.mean(dict_dict['dict_rh3']['celltype_total'][0:2,:],axis = 0),axis=0)
dict_dict['dict_bbc5']['celltype_total'] = np.insert(dict_dict['dict_bbc5']['celltype_total'],[0],np.mean(dict_dict['dict_bbc5']['celltype_total'][0:2,:],axis = 0),axis=0)
dict_dict['dict_rh8']['celltype_total'] = np.insert(dict_dict['dict_rh8']['celltype_total'],[0],np.mean(dict_dict['dict_rh8']['celltype_total'][0:2,:],axis = 0),axis=0)

# Data interpolation (linear)
dict_dict['dict_bc7']['nor_nclus_total'] = interp_session_loss(dict_dict['dict_bc7']['nor_nclus_total'],np.reshape(dict_dict['dict_bc7']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh7']['nor_nclus_total'] = interp_session_loss(dict_dict['dict_rh7']['nor_nclus_total'],np.reshape(dict_dict['dict_rh7']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh3']['nor_nclus_total'] = interp_session_loss(dict_dict['dict_rh3']['nor_nclus_total'],np.reshape(dict_dict['dict_rh3']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_bbc5']['nor_nclus_total'] = interp_session_loss(dict_dict['dict_bbc5']['nor_nclus_total'],np.reshape(dict_dict['dict_bbc5']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh8']['nor_nclus_total'] = interp_session_loss(dict_dict['dict_rh8']['nor_nclus_total'],np.reshape(dict_dict['dict_rh8']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_bc7']['act_nclus'] = interp_session_loss(dict_dict['dict_bc7']['act_nclus'],np.reshape(dict_dict['dict_bc7']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh7']['act_nclus'] = interp_session_loss(dict_dict['dict_rh7']['act_nclus'],np.reshape(dict_dict['dict_rh7']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh3']['act_nclus'] = interp_session_loss(dict_dict['dict_rh3']['act_nclus'],np.reshape(dict_dict['dict_rh3']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_bbc5']['act_nclus'] = interp_session_loss(dict_dict['dict_bbc5']['act_nclus'],np.reshape(dict_dict['dict_bbc5']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh8']['act_nclus'] = interp_session_loss(dict_dict['dict_rh8']['act_nclus'],np.reshape(dict_dict['dict_rh8']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_bc7']['celltype_total'] = interp_session_loss(dict_dict['dict_bc7']['celltype_total'],np.reshape(dict_dict['dict_bc7']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh7']['celltype_total'] = interp_session_loss(dict_dict['dict_rh7']['celltype_total'],np.reshape(dict_dict['dict_rh7']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh3']['celltype_total'] = interp_session_loss(dict_dict['dict_rh3']['celltype_total'],np.reshape(dict_dict['dict_rh3']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_bbc5']['celltype_total'] = interp_session_loss(dict_dict['dict_bbc5']['celltype_total'],np.reshape(dict_dict['dict_bbc5']['x_ticks_labels'],[-1]),day_axis_ideal)
dict_dict['dict_rh8']['celltype_total'] = interp_session_loss(dict_dict['dict_rh8']['celltype_total'],np.reshape(dict_dict['dict_rh8']['x_ticks_labels'],[-1]),day_axis_ideal)

## No response NEURONS (FR no change)
filename_save = os.path.join(output_folder,'noresponse_neurons.png')
# D_less300[0] = dict_dict['dict_rh3']['act_nclus'][:,0]          # shank A is inside the stroked barrel
D_less300[0] = dict_dict['dict_rh3']['nor_nclus_total'][:,1]
D_less300[1] = dict_dict['dict_bc7']['nor_nclus_total'][:,0]
D_less300[2] = dict_dict['dict_rh7']['nor_nclus_total'][:,1]
D_less300[3] = dict_dict['dict_rh7']['nor_nclus_total'][:,2]
D_less300[4] = dict_dict['dict_rh8']['nor_nclus_total'][:,0]            # Shank A is 100 um away from the stroke site. Also its half out
D_less300[5] = dict_dict['dict_rh8']['nor_nclus_total'][:,1]            # Shank B is 350-400um (Qualifies for this bucket)
# D_less300[5] = dict_dict['dict_bbc5']['act_nclus'][:,3]         # shank D is 50 um  (ie too close to the stroked barrel cortex)
D_less300[4] = dict_dict['dict_bbc5']['nor_nclus_total'][:,2]           # Shank C is 250 um 
D_great300[0] = dict_dict['dict_bc7']['nor_nclus_total'][:,2]
# D_great300[1] = dict_dict['dict_bc6']['act_nclus'][:,2]       # BC6 rejected
D_great300[1] = dict_dict['dict_rh7']['nor_nclus_total'][:,0]
outside_barrel[0] = dict_dict['dict_bc7']['nor_nclus_total'][:,1]     # Shank B is outside
outside_barrel[1] = dict_dict['dict_rh3']['nor_nclus_total'][:,2]     # Shank C is outside    (qualifies for outside)
outside_barrel[2] = dict_dict['dict_bbc5']['nor_nclus_total'][:,1]    # shank B is outside (clearly)
S2[0] = dict_dict['dict_bc7']['nor_nclus_total'][:,3]                 # Shank D is in S2
# S2[1] = dict_dict['dict_bc6']['act_nclus'][:,3]               # Shank D of BC6 (BC6 is rejected)
S2[1] = dict_dict['dict_rh7']['nor_nclus_total'][:,3]                 # Shank D is in S2
S2[2] = dict_dict['dict_rh3']['nor_nclus_total'][:,2]                 # Shank C is in S2      (qualifies for S2 as well)
S2[3] = dict_dict['dict_rh8']['nor_nclus_total'][:,2]                 # Shank C is either outside S2 or in S2 (hard to tell). Also this shank is 80% out
S2[4] = dict_dict['dict_rh8']['nor_nclus_total'][:,3]                 # Shank D is in S2 (clear from # of activated neurons)

f, a = plt.subplots(1,1)
a.set_ylabel('Pop. Count')
len_str = 'Population Analysis'
f.suptitle(len_str)
a.set_title("No Response Neurons")
# a.plot(x_ticks_labels,D_less150[0],'r', lw=1.5)
a.plot(x_axis_time,np.sum(list(D_less300.values()),axis=0),'gs--', lw=2.0)
a.plot(x_axis_time,np.sum(list(D_great300.values()),axis=0),'bo--', lw=2.0)
a.plot(x_axis_time,np.sum(list(S2.values()),axis=0),'y>--', lw=2.0)
a.plot(x_axis_time,np.sum(list(outside_barrel.values()),axis=0),'kx--', lw=2.0)
# a.legend(['<150um','150<D<300','D>300', 'S2', 'Outside BC'])
a.legend(['D<300','D>300', 'S2', 'Outside BC'])
f.set_size_inches((10, 6), forward=False)
# labels = [item.get_text() for item in a.get_xticklabels()]
# labels[0] = '-1'
# a.set_xticklabeles(labels)
plt.show()
plt.savefig(filename_save,format = 'png')
plt.close(f)



## ACTIVATED NEURONS (FR goes up)
filename_save = os.path.join(output_folder,'activated_neurons.png')
# D_less300[0] = dict_dict['dict_rh3']['act_nclus'][:,0]          # shank A is inside the stroked barrel
D_less300[0] = dict_dict['dict_rh3']['act_nclus'][:,1]
D_less300[1] = dict_dict['dict_bc7']['act_nclus'][:,0]
D_less300[2] = dict_dict['dict_rh7']['act_nclus'][:,1]
D_less300[3] = dict_dict['dict_rh7']['act_nclus'][:,2]
D_less300[4] = dict_dict['dict_rh8']['act_nclus'][:,0]            # Shank A is 100 um away from the stroke site. Also its half out
D_less300[5] = dict_dict['dict_rh8']['act_nclus'][:,1]            # Shank B is 350-400um (Qualifies for this bucket)
# D_less300[5] = dict_dict['dict_bbc5']['act_nclus'][:,3]         # shank D is 50 um  (ie too close to the stroked barrel cortex)
D_less300[4] = dict_dict['dict_bbc5']['act_nclus'][:,2]           # Shank C is 250 um 
D_great300[0] = dict_dict['dict_bc7']['act_nclus'][:,2]
# D_great300[1] = dict_dict['dict_bc6']['act_nclus'][:,2]       # BC6 rejected
D_great300[1] = dict_dict['dict_rh7']['act_nclus'][:,0]
outside_barrel[0] = dict_dict['dict_bc7']['act_nclus'][:,1]     # Shank B is outside
outside_barrel[1] = dict_dict['dict_rh3']['act_nclus'][:,2]     # Shank C is outside    (qualifies for outside)
outside_barrel[2] = dict_dict['dict_bbc5']['act_nclus'][:,1]    # shank B is outside (clearly)
S2[0] = dict_dict['dict_bc7']['act_nclus'][:,3]                 # Shank D is in S2
# S2[1] = dict_dict['dict_bc6']['act_nclus'][:,3]               # Shank D of BC6 (BC6 is rejected)
S2[1] = dict_dict['dict_rh7']['act_nclus'][:,3]                 # Shank D is in S2
S2[2] = dict_dict['dict_rh3']['act_nclus'][:,2]                 # Shank C is in S2      (qualifies for S2 as well)
S2[3] = dict_dict['dict_rh8']['act_nclus'][:,2]                 # Shank C is either outside S2 or in S2 (hard to tell). Also this shank is 80% out
S2[4] = dict_dict['dict_rh8']['act_nclus'][:,3]                 # Shank D is in S2 (clear from # of activated neurons)

f, a = plt.subplots(1,1)
a.set_ylabel('Pop. Count')
len_str = 'Population Analysis'
f.suptitle(len_str)
a.set_title("Activated Neurons")
# a.plot(x_ticks_labels,D_less150[0],'r', lw=1.5)
a.plot(x_axis_time,np.sum(list(D_less300.values()),axis=0),'gs--', lw=2.0)
a.plot(x_axis_time,np.sum(list(D_great300.values()),axis=0),'bo--', lw=2.0)
a.plot(x_axis_time,np.sum(list(S2.values()),axis=0),'y>--', lw=2.0)
a.plot(x_axis_time,np.sum(list(outside_barrel.values()),axis=0),'kx--', lw=2.0)
# a.legend(['<150um','150<D<300','D>300', 'S2', 'Outside BC'])
a.legend(['D<300','D>300', 'S2', 'Outside BC'])
f.set_size_inches((10, 6), forward=False)
# labels = [item.get_text() for item in a.get_xticklabels()]
# labels[0] = '-1'
# a.set_xticklabeles(labels)
plt.show()
plt.savefig(filename_save,format = 'png')
plt.close(f)

# CELL TYPE ANALYSIS
filename_save = os.path.join(output_folder,'cell_type.png')
cell_total = {}
cell_total[0] = dict_dict['dict_bc7']['celltype_total']
cell_total[1] = dict_dict['dict_rh7']['celltype_total']
cell_total[2] = dict_dict['dict_rh3']['celltype_total']
cell_total[3] = dict_dict['dict_bbc5']['celltype_total']
cell_total[4] = dict_dict['dict_rh8']['celltype_total']
cell_total = np.sum(list(cell_total.values()),axis=0)
# # celltype_arr_rh3 = pop_stats[0]['celltype_total']
# # celltype_arr_rh3 = np.row_stack((celltype_arr_rh3,np.asarray([48,15,8])))
# # celltype_arr_bc7 = pop_stats[1]['celltype_total']
# # celltype = (celltype_arr_rh3 + celltype_arr_bc7)/2
# # bsl_celltype = (celltype[0] + celltype[1])/2
# # celltype = (celltype/bsl_celltype - 1)*100
# # x_ticks_labels = pop_stats[1]['x_ticks_labels']
f, a = plt.subplots(1,1)
a.set_ylabel('Pop. Count')
len_str = 'Population Analysis'
f.suptitle(len_str)
a.set_title("Cell Types")
a.plot(x_axis_time,cell_total[:,0],'g^--', lw=2.0)
a.plot(x_axis_time,cell_total[:,1],'rs--', lw=2.0)
a.plot(x_axis_time,cell_total[:,2],'bo--', lw=2.0)
a.legend(['Pyramidal','Narrow','Wide'])
f.set_size_inches((10, 6), forward=False)
plt.savefig(filename_save,format = 'png')
plt.close(f)








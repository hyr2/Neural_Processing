#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:38:04 2022

@author: hyr2-office
"""

# A poorly written code
# Used to combine animals 
# Must run FR_TS_pop.py for each animal before hand

import scipy.io as sio # Import function to read data.
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import pandas as pd
import seaborn as sns

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
output_folder = '/home/hyr2-office/Documents/Data/NVC/'

mouse_rh3 = os.path.join(mouse_rh3, 'full_mouse_ephys.mat')
mouse_bc7 = os.path.join(mouse_bc7, 'full_mouse_ephys.mat')
mouse_bc6 = os.path.join(mouse_bc6, 'full_mouse_ephys.mat')


pop_stats = {}

pop_stats[0] = sio.loadmat(mouse_rh3)
pop_stats[1] = sio.loadmat(mouse_bc7)
pop_stats[2] = sio.loadmat(mouse_bc6)

D_less150 = {}
D_great150_less300 = {}
D_less300 = {}
D_great300 = {}
outside_barrel = {}
S2 = {}

# inhib
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

# excit
filename_save = os.path.join(output_folder,'activated_neurons.png')
# D_less150[0] = pop_stats[0]['act_nclus'][:,0]           
# D_great150_less300[0] = pop_stats[0]['act_nclus'][:,1]
# D_great150_less300[1] = pop_stats[1]['act_nclus'][:,0]
D_less300[0] = pop_stats[0]['act_nclus'][:,0]           
D_less300[1] = pop_stats[0]['act_nclus'][:,1]
D_less300[2] = pop_stats[1]['act_nclus'][:,0]
D_great300[0] = pop_stats[1]['act_nclus'][:,2]
D_great300[1] = pop_stats[2]['act_nclus'][:,2]
outside_barrel[0] = pop_stats[1]['act_nclus'][:,1] 
outside_barrel[1] = pop_stats[0]['act_nclus'][:,2]
S2[0] = pop_stats[1]['act_nclus'][:,3] 
S2[1] = pop_stats[2]['act_nclus'][:,3] 

# Fixing data
# D_less150[0] = np.append(D_less150[0],D_less150[0][-1])
# D_great150_less300[0] = np.append(D_great150_less300[0],D_great150_less300[0][-1])
D_less300[0] = np.append(D_less300[0],D_less300[0][-1])
D_less300[1] = np.append(D_less300[1],D_less300[1][-1])
D_great300[1] = np.delete(D_great300[1],-2)
D_great300[1][-2] = np.average([D_great300[1][-3],D_great300[1][-1]]) 
outside_barrel[1] = np.append(outside_barrel[1],np.average([outside_barrel[1][-1],outside_barrel[1][-2]]))
S2[1] = np.delete(S2[1],-2)
S2[1][-2] = np.average([S2[1][-3],S2[1][-1]]) 

# Add subplot and layout of figure
x_ticks_labels = pop_stats[1]['x_ticks_labels']
x_axis_time = np.array([-2,-1,2,7,14,21,28,42])
f, a = plt.subplots(1,1)
a.set_ylabel('Pop. Count')
len_str = 'Population Analysis'
f.suptitle(len_str)
a.set_title("Activated Neurons")
# a.plot(x_ticks_labels,D_less150[0],'r', lw=1.5)
a.plot(x_axis_time,np.mean(list(D_less300.values()),axis=0),'gs--', lw=2.0)
a.plot(x_axis_time,np.mean(list(D_great300.values()),axis=0),'bo--', lw=2.0)
a.plot(x_axis_time,np.mean(list(S2.values()),axis=0),'y>--', lw=2.0)
a.plot(x_axis_time,np.mean(list(outside_barrel.values()),axis=0),'kx--', lw=2.0)
# a.legend(['<150um','150<D<300','D>300', 'S2', 'Outside BC'])
a.legend(['D<300','D>300', 'S2', 'Outside BC'])
f.set_size_inches((10, 6), forward=False)
# labels = [item.get_text() for item in a.get_xticklabels()]
# labels[0] = '-1'
# a.set_xticklabeles(labels)
plt.show()
plt.savefig(filename_save,format = 'png')
plt.close(f)

# cell type
# only bc7 and rh3 included
filename_save = os.path.join(output_folder,'cell_type.png')
celltype_arr_rh3 = pop_stats[0]['celltype_total']
celltype_arr_rh3 = np.row_stack((celltype_arr_rh3,np.asarray([48,15,8])))
celltype_arr_bc7 = pop_stats[1]['celltype_total']
celltype = (celltype_arr_rh3 + celltype_arr_bc7)/2
bsl_celltype = (celltype[0] + celltype[1])/2
celltype = (celltype/bsl_celltype - 1)*100
x_ticks_labels = pop_stats[1]['x_ticks_labels']
f, a = plt.subplots(1,1)
a.set_ylabel('Pop. Count')
len_str = 'Population Analysis'
f.suptitle(len_str)
a.set_title("Cell Types")
a.plot(x_axis_time,celltype[:,0],'g^--', lw=2.0)
a.plot(x_axis_time,celltype[:,1],'rs--', lw=2.0)
a.plot(x_axis_time,celltype[:,2],'bo--', lw=2.0)
a.legend(['Pyramidal','Narrow','Wide'])
f.set_size_inches((10, 6), forward=False)
plt.savefig(filename_save,format = 'png')
plt.close(f)







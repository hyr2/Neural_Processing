#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 00:16:22 2023

@author: hyr2-office
"""

# This is an impedence script. Extract impedences and then plots then
# over a longitudinal period

import os, gc, warnings, json, sys, glob, shutil
from copy import deepcopy
sys.path.append(os.path.join(os.getcwd(),'Intan-File-Reader'))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from load_intan_rhd_format import read_data, read_header
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting fonts
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=16)          # controls default text sizes
sns.set(font_scale=1)

input_dir = '/home/hyr2-office/Documents/Data/NVC/RH-9_new/'

file_list = os.listdir(input_dir)
file_list = natsorted(file_list)

df_single_animal = pd.DataFrame({
    "day" : [],
    "Z" : [],
    }
    )

Days_arr = np.array([1,2,3,4,6,10,22,29,36,43,50])  # RH9
# Days_arr = np.array([1,8,11,14,19,26,36,40,47,54,61,68])  # RH7
# Days_arr = np.array([1,3,5,8,13,20,27,34,41,48,55,62])  # RH8
Days_arr = Days_arr + 28
for iter_local,iter_name in enumerate(file_list):
    print(iter_name)
    with open(os.path.join(input_dir, iter_name), "rb") as fh:
        head_dict = read_header(fh)
    tmp1 = head_dict['amplifier_channels']
    curr_day = Days_arr[iter_local]
    for iter_1 in range(len(tmp1)):
        Z_curr = tmp1[iter_1]['electrode_impedance_magnitude'] / 1e3                     # KOhms
        if Z_curr < 3500:
            df_single_animal.loc[len(df_single_animal.index)] = [curr_day,Z_curr]



fig, axes = plt.subplots(1,1, figsize=(6,9), dpi=200)
# # axes = axes.flatten()
# for i in Days_arr:
#     y = df_single_animal[df_single_animal.day == i].dropna()
#     # Add some random "jitter" to the x-axis
#     x = np.random.normal(i, 0.04, size=len(y))
#     axes.plot(x, y, 'r.', alpha=0.2)
# df_single_animal.boxplot(by = 'day',showmeans=True, meanprops={'marker':'x','markerfacecolor':'black', 'markeredgecolor':'black'},ax = axes)

filename_save = os.path.join(input_dir,'imped_summary.png')
ax = sns.boxplot(x="day", y="Z", data=df_single_animal, showfliers = False,meanline = True, meanprops={'marker':'x','markerfacecolor':'black', 'markeredgecolor':'black'})
ax = sns.swarmplot(x="day", y="Z", data=df_single_animal, color="black", alpha = 0.6, edgecolor = 'b')
ax.set_ylim(0,3.5e3)
ax.set_xlabel('Days Post Surgery')
ax.set_ylabel('Impedence (kOhms)')
ax.set_title('RH9')
fig.set_size_inches((9, 6), forward=False)
plt.savefig(filename_save,format = 'png')
plt.close()


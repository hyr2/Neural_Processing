# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:37:55 2023

@author: haadr
"""

import time, pickle
from scipy.io import loadmat
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from copy import deepcopy
from sklearn.neighbors import KernelDensity

input_folder = r'C:\Rice_2023\Data\Results\IOS_results_10_18_2023'
data_file = os.path.join(input_folder,'output_data.mat')
D1 = loadmat(data_file)

t = time.localtime()
current_time = time.strftime("%m_%d_%Y_%H_%M", t)
output_folder = os.path.join(r'C:\Rice_2023\Data\Results',current_time)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

arr_area_all = D1['arr_area_all']
xaxis_ideal = np.squeeze(D1['xaxis_ideal'])
shift_vec_chronic =D1['shift_vec_chronic'] 
shift_vec_early = D1['shift_vec_early']

# For the area of activation (averaged over animals n=7)
arr_area_mean = np.nanmean(arr_area_all,axis = 0)
arr_area_count = np.count_nonzero(~np.isnan(arr_area_all),axis=0)
arr_area_sem = np.nanstd(arr_area_all,axis =0)/np.sqrt(arr_area_count)
# For the histogram of shifts in cortical activation area (chronic vs subacute/recovery)
df_shifts1 = pd.DataFrame(np.transpose(-shift_vec_chronic),columns = ['shift'])
col_tmp = ['chro']*np.shape(shift_vec_chronic)[1]
df_shifts1['phase'] = col_tmp
df_shifts2 = pd.DataFrame(np.transpose(shift_vec_early),columns = ['shift'])
col_tmp = ['rec']*np.shape(shift_vec_early)[1]
df_shifts2['phase'] = col_tmp
df_shifts = pd.concat((df_shifts1,df_shifts2),axis = 0)


fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
# axes = axes.flatten()
fig.tight_layout(pad = 5)
axes.errorbar(xaxis_ideal,arr_area_mean,yerr = arr_area_sem, capsize=2, \
                 linewidth = 4.5,color = 'k',ecolor = '#B2BEB5',elinewidth = 2.5, \
                 marker = 'o',markeredgecolor = 'k',markeredgewidth = 1, markersize = 10, \
                 markerfacecolor = '#B2BEB5')
axes.set_ylabel('Activation Area (relative)',fontsize = 20)
axes.set_xlabel('Days Post Stroke',fontsize=20)
sns.despine()
fig.set_size_inches(7,4,forward=True)
fig.savefig(os.path.join(output_folder,'Area_longitudinal_IOS.png'),format='png',dpi = 300)
fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
# axes = axes.flatten()
fig.tight_layout(pad = 5)
sns.histplot(df_shifts,x = 'shift',hue='phase',element = 'step',binwidth=0.25,ax = axes,palette = ['#B2BEB5','k'])
axes.set_xlabel('Inter-day Shift(mm)',fontsize = 20)
axes.set_ylabel('histogram',fontsize = 20)
sns.despine()
fig.set_size_inches(6,3.5,forward=True)
fig.savefig(os.path.join(output_folder,'Histogram_shifts_IOS.png'),format='png',dpi = 300)


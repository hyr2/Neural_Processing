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

# input_folder = r'C:\Rice_2023\Data\Results\IOS_results_10_18_2023'
input_folder = '/home/hyr2-office/Documents/Data/IOS_imaging'
data_file = os.path.join(input_folder,'output_data.mat')
D1 = loadmat(data_file)

t = time.localtime()
current_time = time.strftime("%m_%d_%Y_%H_%M", t)
# output_folder = os.path.join(r'C:\Rice_2023\Data\Results',current_time)
output_folder = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

arr_area_all = D1['arr_area_all']
xaxis_ideal = np.squeeze(D1['xaxis_ideal'])
shift_vec_chronic =D1['shift_vec_chronic'] 
shift_vec_early = D1['shift_vec_early']
shift_vec_days = np.squeeze(D1['avg_shift_by_day'])
shift_vec_days_sem = np.squeeze(D1['sem_shift_by_day'])
xaxis_ideal_1h = np.squeeze(D1['day_axis'])
chr_distances = D1['chr_distances']
bsl_distances = D1['bsl_distances']

# For cortical activity shift vs spiking activity shift 1I
fig, axes = plt.subplots(1,1, figsize=(2,1.5), dpi=300)
df_shifts1 = pd.DataFrame(np.transpose(bsl_distances),columns = ['dist'])
col_tmp = ['bsl']*np.shape(bsl_distances)[1]
df_shifts1['phase'] = col_tmp
df_shifts2 = pd.DataFrame(np.transpose(chr_distances),columns = ['dist'])
col_tmp = ['chr']*np.shape(chr_distances)[1]
df_shifts2['phase'] = col_tmp
df_shifts = pd.concat((df_shifts1,df_shifts2),axis = 0)
sns.barplot(data=df_shifts,y = 'dist',x = 'phase',orient = 'v',color = '#A8A9A8', \
            linewidth=2.5, edgecolor="k",ax = axes)
fig.set_size_inches(1,1.5,forward=True)
sns.despine()
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True)         # ticks along the top edge are off
axes.set_ylim(0,1.1)
axes.set_yticks([])
fig.savefig(os.path.join(output_folder,'CorticalMapVsSpiking.png'),format='png',dpi = 300)
print('mean distance between modalities pre-stroke: ', np.mean(bsl_distances) )
print('sem distance between modalities pre-stroke: ', np.std(bsl_distances)/bsl_distances.shape[1])
print('mean distance between modalities post-stroke', np.mean(chr_distances) )
print('sem distance between modalities pre-stroke: ', np.std(chr_distances)/chr_distances.shape[1])

# For 1H
fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=300)
axes.errorbar(xaxis_ideal_1h,shift_vec_days,yerr = shift_vec_days_sem, capsize=2, \
                 linewidth = 4.5,color = 'k',ecolor = '#B2BEB5',elinewidth = 2.5, \
                 marker = 'o',markeredgecolor = 'k',markeredgewidth = 1, markersize = 8, \
                 markerfacecolor = '#B2BEB5')
axes.set_ylabel('Shift (mm)',fontsize = 20)
axes.set_xlabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True)         # ticks along the top edge are off

sns.despine(ax=axes, left=False, bottom=False, trim=False)
axes.set_ylim(0,0.9)
axes.set_yticks([0,0.5])
fig.set_size_inches(4,3,forward=True)
fig.savefig(os.path.join(output_folder,'1H_shift_IOS.png'),format='png',dpi = 300)
print('Avg shift first 3 datapoints: ',np.mean(shift_vec_days[0:3]))
print('Avg shift last 3 datapoints: ',np.mean(shift_vec_days[-3:]))
tmp_avg_sem_3_datapoints = np.square(shift_vec_days_sem)/9
print('sem shift first 3 datapoints: ',np.sqrt(np.sum(tmp_avg_sem_3_datapoints[0:3])))
print('sem shift last 3 datapoints: ',np.sqrt(np.sum(tmp_avg_sem_3_datapoints[-3:])))

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


fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=300)
axes.errorbar(xaxis_ideal,arr_area_mean,yerr = arr_area_sem, capsize=2, \
                 linewidth = 4.5,color = 'k',ecolor = '#B2BEB5',elinewidth = 2.5, \
                 marker = 'o',markeredgecolor = 'k',markeredgewidth = 1, markersize = 8, \
                 markerfacecolor = '#B2BEB5')
axes.set_ylabel('Activation Area (relative)',fontsize = 20)
axes.set_xlabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True)         # ticks along the top edge are off

sns.despine(ax=axes, left=False, bottom=False, trim=False)
axes.set_ylim(0,1.1)
axes.set_yticks([0,0.5,1])
fig.set_size_inches(4,3,forward=True)
fig.savefig(os.path.join(output_folder,'Area_longitudinal_IOS.png'),format='png',dpi = 300)

# Histograms
# Histograms
fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=300)
# axes = axes.flatten()
#fig.tight_layout(pad = 5)
sns.histplot(df_shifts,x = 'shift',hue='phase',element = 'step',binwidth=0.25,ax = axes,palette = ['#A8A9A8','k'])
# axes.set_xlabel('Inter-day Shift(mm)',fontsize = 20)
axes.set_ylabel('histogram',fontsize = 20)
axes.set_xlabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True,
    labelright=False)         # ticks along the top edge are off
fig.set_size_inches(6,4,forward=True)
sns.despine(ax=axes, left=False, bottom=False, trim=False,offset = 2.5)
fig.savefig(os.path.join(output_folder,'Histogram_shifts_IOS.png'),format='png',dpi = 300)

# Another histogram 
# For the histogram of shifts in cortical activation area (chronic vs subacute/recovery)
df_shifts1 = pd.DataFrame(np.transpose(shift_vec_chronic),columns = ['shift'])
col_tmp = ['chro']*np.shape(shift_vec_chronic)[1]
df_shifts1['phase'] = col_tmp
df_shifts2 = pd.DataFrame(np.transpose(shift_vec_early),columns = ['shift'])
col_tmp = ['rec']*np.shape(shift_vec_early)[1]
df_shifts2['phase'] = col_tmp
df_shifts = pd.concat((df_shifts1,df_shifts2),axis = 0)

fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=300)
sns.histplot(df_shifts,x = 'shift',hue='phase',element = 'step',binwidth=0.175,ax = axes,palette = ['#AB5454','#54ABAB'],alpha=0.1,linewidth = 3.25)
axes.legend_.remove()
axes.set_ylabel('histogram',fontsize = 20)
axes.set_xlabel('')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True,
    labelright=False)         # ticks along the top edge are off
fig.set_size_inches(6,4,forward=True)
sns.despine(ax=axes, left=False, bottom=False, trim=False,offset = 2.5)
axes.set_xticks([0,0.75,1.5])
axes.set_yticks([0,5,10,15])
fig.savefig(os.path.join(output_folder,'Histogram_shifts_IOS_LL.png'),format='png',dpi = 300)


# Specify the bin width and calculate the number of bins
fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=300)
fig.tight_layout(pad = 5)
bin_width = 0.25    # in mm

num_bins = int((np.amax(shift_vec_early) - np.amin(shift_vec_early)) / bin_width) + 1
bin_edges = [0 + i * bin_width for i in range(num_bins + 1)] # Define the bin edges
hist, bin_edges = np.histogram(shift_vec_early, bins=bin_edges)
plt.bar(bin_edges[:-1],hist,width =bin_width,align='edge',alpha=1,color='k')

num_bins = int((np.amax(shift_vec_chronic) - np.amin(shift_vec_chronic)) / bin_width) + 1
bin_edges = [0 + i * bin_width for i in range(num_bins + 1)] # Define the bin edges
hist, bin_edges = np.histogram(shift_vec_chronic, bins=bin_edges)
plt.bar(bin_edges[:-1],hist,width = bin_width,align='edge',alpha=0.35,color='#A8A9A8',edgecolor='#616261')
axes.set_xlabel('')
axes.set_xticks([0,0.5,1,1.5])
axes.set_yticks([5,10,15])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True,
    labelright=True)         # ticks along the top edge are off
fig.set_size_inches(6,4,forward=True)
sns.despine(ax=axes, left=False, bottom=False, trim=False,offset = 2.5)
fig.set_size_inches(6,4,forward=True)
fig.savefig(os.path.join(output_folder,'Histogram_shifts_IOS_1.png'),format='png',dpi = 300)

# Specify the bin width and calculate the number of bins
fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=300)
fig.tight_layout(pad = 5)
bin_width = 0.25    # in mm

num_bins = int((np.amax(shift_vec_early) - np.amin(shift_vec_early)) / bin_width) + 1
bin_edges = [0 + i * bin_width for i in range(num_bins + 1)] # Define the bin edges
hist, bin_edges = np.histogram(shift_vec_early, bins=bin_edges)
plt.bar(bin_edges[:-1],hist,width =bin_width,align='edge',alpha=1,color='k')

num_bins = int((np.amax(shift_vec_chronic) - np.amin(shift_vec_chronic)) / bin_width) + 1
bin_edges = [0 + i * bin_width for i in range(num_bins + 1)] # Define the bin edges
hist, bin_edges = np.histogram(-shift_vec_chronic, bins=np.flip(-np.array(bin_edges)))
plt.bar(bin_edges[:-1],hist,width = bin_width,align='edge',alpha=1,color='#A8A9A8',edgecolor='#616261')
axes.set_xlabel('')
sns.despine(ax=axes, left=False, bottom=False, trim=False,offset=10)
axes.tick_params(axis='y', right=False) 
axes.set_yticks([5,10,15])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True,
    labelright=True)         # ticks along the top edge are off

sns.despine(ax=axes, left=False, bottom=False, trim=False,offset = 2.5)
fig.set_size_inches(6,4,forward=True)
# axes.set_xticks([-1,0,1])
plt.xticks([-1,0,1], ['1','0','1'])


fig.savefig(os.path.join(output_folder,'Histogram_shifts_IOS_2.png'),format='png',dpi = 300)

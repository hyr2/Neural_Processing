#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 20:50:56 2023

@author: hyr2-office
"""

import time
import scipy.stats as sstats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import pandas as pd
import seaborn as sns
from copy import deepcopy
import sklearn.cluster as Clustering


# Plotting fonts
sns.set_style('white') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=16)          # controls default text sizes

# from stat_tests_support import stat_test_spikes

df_all_clusters_main = pd.DataFrame(
    {
        "animal_ID":[],
        "day": [],
        "shank": [],
        "celltype": [],
        "response": [],
        "T2P": [],
        "spont_FR": [],
        "event_FR": [],
        "burst_i": [],
        "tau_r": [],
        "wav_assym": []
    }
)


mouse_rh3 = os.path.join('/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh3/','all_cluster.pkl')
mouse_bc7 = os.path.join('/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_bc7/','all_cluster.pkl')
# mouse_bc6 = os.path.join('/home/hyr2-office/Documents/Data/NVC/BC6/','all_cluster.pkl')
mouse_rh7 = os.path.join('/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh7/','all_cluster.pkl')
mouse_bbc5 = os.path.join('/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_bbc5/','all_cluster.pkl')
# mouse_bc8 = os.path.join('/home/hyr2-office/Documents/Data/NVC/BC8/','all_cluster.pkl')
mouse_rh8 = os.path.join('/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh8/','all_cluster.pkl')
mouse_rh9 = os.path.join('/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh9/','all_cluster.pkl')

t = time.localtime()
current_time = time.strftime("%m_%d_%Y_%H_%M", t)
output_folder = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


df_tmp = pd.read_pickle(mouse_rh3)
df_all_clusters_main = pd.concat([df_all_clusters_main,df_tmp],axis = 0)
df_tmp = pd.read_pickle(mouse_rh8)
df_all_clusters_main = pd.concat([df_all_clusters_main,df_tmp],axis = 0)
df_tmp = pd.read_pickle(mouse_rh9)
df_all_clusters_main = pd.concat([df_all_clusters_main,df_tmp],axis = 0)
df_tmp = pd.read_pickle(mouse_rh7)
df_all_clusters_main = pd.concat([df_all_clusters_main,df_tmp],axis = 0)
df_tmp = pd.read_pickle(mouse_bc7)
df_all_clusters_main = pd.concat([df_all_clusters_main,df_tmp],axis = 0)
df_tmp = pd.read_pickle(mouse_bbc5)
df_all_clusters_main = pd.concat([df_all_clusters_main,df_tmp],axis = 0)

# Scatter plots baseline
df_all_clusters_main_bsl = df_all_clusters_main[ (df_all_clusters_main['day'] == -3) | (df_all_clusters_main['day'] == -2) ]
df_all_clusters_main_bsl = df_all_clusters_main_bsl[ (df_all_clusters_main_bsl['spont_FR'] > 2) ]
df_celltype = df_all_clusters_main_bsl.groupby('celltype')
fig, axes = plt.subplots(1,3, figsize=(10,12), dpi=100)
fig.tight_layout(pad = 5)
df_I = df_celltype.get_group('NI')                  # All cells by type
sns.scatterplot(df_I,x = 'spont_FR',y = 'event_FR',c = '#0032ff',ax = axes[0])
axes[0].set_aspect('equal','box')
axes[0].set_xlim(0,30)
axes[0].set_ylim(0,46)
df_E = df_celltype.get_group('P')                  # All cells by type
sns.scatterplot(df_E,x = 'spont_FR',y = 'event_FR',c = '#fa3232',ax = axes[1])
axes[1].set_aspect('equal','box')
axes[1].set_xlim(0,30)
axes[1].set_ylim(0,45)
df_WI = df_celltype.get_group('WI')                  # All cells by type
sns.scatterplot(df_WI,x = 'spont_FR',y = 'event_FR',c = '#64c864',ax = axes[2])
axes[2].set_aspect('equal','box')
axes[2].set_xlim(0,25)
axes[2].set_ylim(0,30)
fig.savefig(os.path.join(output_folder,'baseline_scatters_celltype.svg'),format = 'svg')

# scatter recovery phase (days:7,14)
df_all_clusters_main_bsl = df_all_clusters_main[ (df_all_clusters_main['day'] == 7) | (df_all_clusters_main['day'] == 14) ]
df_all_clusters_main_bsl = df_all_clusters_main_bsl[ (df_all_clusters_main_bsl['spont_FR'] > 2) ]
df_celltype = df_all_clusters_main_bsl.groupby('celltype')
fig, axes = plt.subplots(1,3, figsize=(10,12), dpi=100)
fig.tight_layout(pad = 5)
df_I = df_celltype.get_group('NI')                  # All cells by type
sns.scatterplot(df_I,x = 'spont_FR',y = 'event_FR',c = '#0032ff',ax = axes[0])
axes[0].set_aspect('equal','box')
axes[0].set_xlim(0,30)
axes[0].set_ylim(0,46)
df_E = df_celltype.get_group('P')                  # All cells by type
sns.scatterplot(df_E,x = 'spont_FR',y = 'event_FR',c = '#fa3232',ax = axes[1])
axes[1].set_aspect('equal','box')
axes[1].set_xlim(0,30)
axes[1].set_ylim(0,45)
df_WI = df_celltype.get_group('WI')                  # All cells by type
sns.scatterplot(df_WI,x = 'spont_FR',y = 'event_FR',c = '#64c864',ax = axes[2])
axes[2].set_aspect('equal','box')
axes[2].set_xlim(0,25)
axes[2].set_ylim(0,30)
fig.savefig(os.path.join(output_folder,'recovery_scatters_celltype.svg'),format = 'svg')

# scatter chronic phase (days:28,42)
df_all_clusters_main_bsl = df_all_clusters_main[ (df_all_clusters_main['day'] == 28) | (df_all_clusters_main['day'] == 42) ]
df_all_clusters_main_bsl = df_all_clusters_main_bsl[ (df_all_clusters_main_bsl['spont_FR'] > 2) ]
df_celltype = df_all_clusters_main_bsl.groupby('celltype')
fig, axes = plt.subplots(1,3, figsize=(10,12), dpi=100)
fig.tight_layout(pad = 5)
df_I = df_celltype.get_group('NI')                  # All cells by type
sns.scatterplot(df_I,x = 'spont_FR',y = 'event_FR',c = '#0032ff',ax = axes[0])
axes[0].set_aspect('equal','box')
axes[0].set_xlim(0,30)
axes[0].set_ylim(0,46)
df_E = df_celltype.get_group('P')                  # All cells by type
sns.scatterplot(df_E,x = 'spont_FR',y = 'event_FR',c = '#fa3232',ax = axes[1])
axes[1].set_aspect('equal','box')
axes[1].set_xlim(0,30)
axes[1].set_ylim(0,45)
df_WI = df_celltype.get_group('WI')                  # All cells by type
sns.scatterplot(df_WI,x = 'spont_FR',y = 'event_FR',c = '#64c864',ax = axes[2])
axes[2].set_aspect('equal','box')
axes[2].set_xlim(0,25)
axes[2].set_ylim(0,30)
fig.savefig(os.path.join(output_folder,'chronic_scatters_celltype.svg'),format = 'svg')


df_I_act = df_I.groupby('response').get_group(1)    # Only stimulus locked
df_E_act = df_E.groupby('response').get_group(1)
df_WI_act = df_WI.groupby('response').get_group(1)
sns.scatterplot(df_I_act,x = 'spont_FR',y = 'event_FR')
sns.scatterplot(df_E_act,x = 'spont_FR',y = 'event_FR')

tmp_event_FR_I = df_I_act.event_FR.to_numpy()       # Statistical testing
tmp_event_FR_E = df_E_act.event_FR.to_numpy()
result_test = sstats.ranksums(tmp_event_FR_I,tmp_event_FR_E)
tmp_event_FR_I = df_I_act.spont_FR.to_numpy()
tmp_event_FR_E = df_E_act.spont_FR.to_numpy()
result_test = sstats.ranksums(tmp_event_FR_I,tmp_event_FR_E)

sns.scatterplot(df_all_clusters_main_bsl,x = 'burst_i',y = 'wav_assym')

# Histogram of marginals
sns.histplot(data=df_all_clusters_main_bsl, x="T2P", color="red", label="Trough to Peak", kde=True,kde_kws = {'bw_adjust' : 1.3}, binwidth = 0.05)

## Clustering here
# class_clustering = Clustering.AgglomerativeClustering(n_clusters=3).fit(nn_data)
# class_clustering.labels_

# Spontaenous FR by celltype (Fig 1 E)
df_all_clusters_main_all = df_all_clusters_main.groupby(['day','celltype'])
arr_all = df_all_clusters_main_all['spont_FR'].mean().to_numpy()
arr_all = np.reshape(arr_all,[13,3])
arr_all = np.delete(arr_all,[6,8,10,11],axis = 0)
x_axis = np.array([-3,-2,2,7,14,21,28,42,56])
plt.plot(x_axis,arr_all[:,0],'b-',linewidth = 2.5)
plt.plot(x_axis,arr_all[:,1],'r-',linewidth = 2.5)
plt.plot(x_axis,arr_all[:,2],'g-',linewidth = 2.5)
plt.legend(['FS/PV','Pyr','Wide-Inter'])
plt.xlabel('Days')
# plt.savefig(os.path.join('/home/hyr2-office/Documents/Paper/Single-Figures-SVG/Fig1_updated/subfigures/','FR_plot_celltype.svg'),format = 'svg')


# Plotting of Burst Index longitudinal (no spatial info)
df_all_clusters_main.loc[df_all_clusters_main['day'] == 47,'day'] = 49
df_all_clusters_main.loc[df_all_clusters_main['day'] == 24,'day'] = 21
filename_save = os.path.join(output_folder,'burst_index_BOX.svg')
df_all_clusters_main_high = df_all_clusters_main[df_all_clusters_main['spont_FR'] > 1]            # only high spike count clusters considered for this analysis
df_all_clusters_main_high_bsl = df_all_clusters_main_high[(df_all_clusters_main_high['day'] == -3) |(df_all_clusters_main_high['day'] == -2) ]
df_all_clusters_main_high_bsl.loc[:,'day'] = 'Pre'
df_all_clusters_main_high_recovery = df_all_clusters_main_high[(df_all_clusters_main_high['day'] == 7) |(df_all_clusters_main_high['day'] == 14) ]
df_all_clusters_main_high_recovery.loc[:,'day'] = 'Recovery'
df_all_clusters_main_high_chronic = df_all_clusters_main_high[(df_all_clusters_main_high['day'] == 28) |(df_all_clusters_main_high['day'] == 42) ]
df_all_clusters_main_high_chronic.loc[:,'day'] = 'Chronic'
df = pd.concat([df_all_clusters_main_high_bsl,df_all_clusters_main_high_recovery,df_all_clusters_main_high_chronic],axis = 0)
fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
sns.lineplot(x='day', y='burst_i', data = df, linewidth=2,errorbar = 'se',ax = axes,err_style = 'bars')# ax.set_ylim([-1.1,1.5])
axes.set(xlabel=None, ylabel = None,title = 'Network Bursting')
axes.grid(alpha=0.5)
fig.set_size_inches((10, 6), forward=True)
plt.savefig(filename_save,format = 'svg')

# Here Generating FR of all clusters and Population of all clusters 
filename_save = os.path.join(output_folder,'avg_FR_all_clusters.svg')
y_axis_data = df_all_clusters_main.groupby('day').mean()['spont_FR'].to_numpy()[:-1]
y_axis_sem = df_all_clusters_main.groupby('day').sem()['spont_FR'].to_numpy()[:-1]
day_axis = df_all_clusters_main.groupby('day').mean().index.to_numpy()[:-1]
plt.errorbar(x = day_axis,y = y_axis_data,yerr = y_axis_sem,lw = 2.9,color = 'r')
plt.grid()
plt.title('Avg Spontaneous Firing Rate (all Clusters)')
plt.savefig(filename_save,format = 'svg')
plt.figure()
filename_save = os.path.join(output_folder,'avg_pop_all_clusters.svg')
y_pop_ = df_all_clusters_main.groupby('day').count()['spont_FR'].to_numpy()[:-1]
plt.plot(day_axis,y_pop_)
plt.title('Avg Population (all Clusters)')
plt.savefig(filename_save,format = 'svg')
plt.close()


# Bursting Index by E and I cells
plt.figure()
filename_save = os.path.join(output_folder,'burst_index_alldays.svg')
# y = df_all_clusters_main_high.groupby(['day','celltype']).mean()['burst_i'].groupby('celltype').get_group('WI').to_numpy()
# y_e = df_all_clusters_main_high.groupby(['day','celltype']).sem()['burst_i'].groupby('celltype').get_group('WI').to_numpy()
# plt.errorbar(x = np.sort(df_all_clusters_main_high['day'].unique()), y= y, yerr  = y_e , color = 'g',lw = 2.9)
y = df_all_clusters_main_high.groupby(['day','celltype']).mean()['burst_i'].groupby('celltype').get_group('NI').to_numpy()
y_e = df_all_clusters_main_high.groupby(['day','celltype']).sem()['burst_i'].groupby('celltype').get_group('NI').to_numpy()
plt.errorbar(x = np.sort(df_all_clusters_main_high['day'].unique()), y= y, yerr  = y_e , color = 'b',lw = 2.9)
y = df_all_clusters_main_high.groupby(['day','celltype']).mean()['burst_i'].groupby('celltype').get_group('P').to_numpy()
y_e = df_all_clusters_main_high.groupby(['day','celltype']).sem()['burst_i'].groupby('celltype').get_group('P').to_numpy()
plt.errorbar(x = np.sort(df_all_clusters_main_high['day'].unique()), y= y, yerr  = y_e , color = 'r',lw = 2.9)
plt.legend(['FS/PV','Pyr'])
plt.savefig(filename_save,format = 'svg')

df_local = df_all_clusters_main_high.groupby('celltype').get_group('P')[['day','burst_i']]  # for pyramidal
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 2)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 7)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 14)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 21)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 28)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 35)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 42)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 49)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 56)]['burst_i'] )
df_local = df_all_clusters_main_high.groupby('celltype').get_group('NI')[['day','burst_i']]     # FS/PV cells
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 2)]['burst_i'] ) 
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 7)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 14)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 21)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 28)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 35)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 42)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 49)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 56)]['burst_i'] )
df_local = df_all_clusters_main_high.groupby('celltype').get_group('WI')[['day','burst_i']]      # Wide interneurons
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 2)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 7)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 14)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 21)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 28)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 35)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 42)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 49)]['burst_i'] )
sstats.ranksums(df_local[(df_local['day'] == -2)| (df_local['day'] == -3)]['burst_i'], df_local[(df_local['day'] == 56)]['burst_i'] )







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:53:37 2023

@author: hyr2
"""

import os, json
import sys
sys.path.append('utils')
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy import integrate
from scipy.stats import ttest_ind, zscore, norm
from itertools import product, compress
import pandas as pd
from utils.read_mda import readmda
from utils.read_stimtxt import read_stimtxt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns

# Save dir
dir_save = '/home/hyr2-office/Documents/Data/NVC/Results/Tracking-Results/'
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

# script to be run after func_pop_analysis has been run

dir1 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_0/Processed/count_analysis'
dir2 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_1/Processed/count_analysis'
dir3 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_3/Processed/count_analysis'
dir4 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh3/Shank_2/Processed/count_analysis'
dir5 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_bc7/Shank_2/Processed/count_analysis'
dir6 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_bc7/Shank_3/Processed/count_analysis'
dir7 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh8/Shank_1/Processed/count_analysis'
dir8 = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh8/Shank_3/Processed/count_analysis'


file_load1 = os.path.join(dir1,'all_clus_property.npy')
file_load2 = os.path.join(dir2,'all_clus_property.npy')
file_load3 = os.path.join(dir3,'all_clus_property.npy')
file_load4 = os.path.join(dir4,'all_clus_property.npy')
file_load5 = os.path.join(dir5,'all_clus_property.npy')
file_load6 = os.path.join(dir6,'all_clus_property.npy')
file_load7 = os.path.join(dir7,'all_clus_property.npy')
file_load8 = os.path.join(dir8,'all_clus_property.npy')
f1 = np.load(file_load1,allow_pickle=True)
f2 = np.load(file_load2,allow_pickle=True)
f3 = np.load(file_load3,allow_pickle=True)
f4 = np.load(file_load4,allow_pickle=True)
f5 = np.load(file_load5,allow_pickle=True)
f6 = np.load(file_load6,allow_pickle=True)
f7 = np.load(file_load7,allow_pickle=True)
f8 = np.load(file_load8,allow_pickle=True)

main_df1 = pd.DataFrame()
main_df2 = pd.DataFrame()
main_df3 = pd.DataFrame()
main_df4 = pd.DataFrame()
main_df5 = pd.DataFrame()
main_df6 = pd.DataFrame()
main_df7 = pd.DataFrame()
main_df8 = pd.DataFrame()

for iter in range(f1.size):
    tmp_df = pd.DataFrame(f1[iter])
    main_df1 = pd.concat([main_df1,tmp_df],axis = 0)
    
for iter in range(f2.size):
    tmp_df = pd.DataFrame(f2[iter])
    main_df2 = pd.concat([main_df2,tmp_df],axis = 0)
    
for iter in range(f3.size):
    tmp_df = pd.DataFrame(f3[iter])
    main_df3 = pd.concat([main_df3,tmp_df],axis = 0)

for iter in range(f4.size):
    tmp_df = pd.DataFrame(f4[iter])
    main_df4 = pd.concat([main_df4,tmp_df],axis = 0)
    
for iter in range(f5.size):
    tmp_df = pd.DataFrame(f5[iter])
    main_df5 = pd.concat([main_df5,tmp_df],axis = 0)
    
for iter in range(f6.size):
    tmp_df = pd.DataFrame(f6[iter])
    main_df6 = pd.concat([main_df6,tmp_df],axis = 0)
    
for iter in range(f7.size):
    tmp_df = pd.DataFrame(f7[iter])
    main_df7 = pd.concat([main_df7,tmp_df],axis = 0)

for iter in range(f8.size):
    tmp_df = pd.DataFrame(f8[iter])
    main_df8 = pd.concat([main_df8,tmp_df],axis = 0)

# fixing cluster IDs    
main_df2['cluster_id'] = main_df2['cluster_id'] + main_df1['cluster_id'].max()
main_df3['cluster_id'] = main_df3['cluster_id'] + main_df2['cluster_id'].max()
main_df4['cluster_id'] = main_df4['cluster_id'] + main_df3['cluster_id'].max()
main_df5['cluster_id'] = main_df5['cluster_id'] + main_df4['cluster_id'].max()
main_df6['cluster_id'] = main_df6['cluster_id'] + main_df5['cluster_id'].max()
main_df7['cluster_id'] = main_df7['cluster_id'] + main_df6['cluster_id'].max()
main_df8['cluster_id'] = main_df8['cluster_id'] + main_df7['cluster_id'].max()

main_df = pd.concat([main_df1,main_df2,main_df3,main_df4,main_df5,main_df6,main_df7,main_df8],axis = 0)


# Plot for : Populational: neuron numbers of increased/lowered firing 
main_df_counts = main_df.filter(['cluster_id','plasticity_metric'])
main_df_counts = main_df_counts.drop_duplicates()

data = [(main_df_counts['plasticity_metric'] == 1).sum(), (main_df_counts['plasticity_metric'] == 0).sum(), (main_df_counts['plasticity_metric'] == -1).sum()]

ax = sns.countplot(x = 'plasticity_metric', data = main_df_counts,palette = ['r','#6F8FAF','g'])
# ax.set_yticks([0,5,10,15,20])
plt.savefig(os.path.join(dir_save,'Plasticity_counts.png'),format = 'png',dpi = 300)

plt.figure()
labels1 = ['Up-regulated','Recovered','Down-regulated']
plt.pie(data,labels=labels1)
plt.savefig(os.path.join(dir_save,'Plasticity_counts_pie.png'),format = 'png',dpi = 300)


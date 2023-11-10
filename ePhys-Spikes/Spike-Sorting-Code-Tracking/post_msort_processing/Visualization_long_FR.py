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

# script to be run after func_pop_analysis has been run

dir1 = '/home/hyr2/Documents/Data/NVC/Results/Tracking-Results-RH11-Twoshanks/Processed_0/count_analysis/'
dir2 = '/home/hyr2/Documents/Data/NVC/Results/Tracking-Results-RH11-Twoshanks/Processed_1/count_analysis/'

file_load1 = os.path.join(dir1,'all_clus_property.npy')
file_load2 = os.path.join(dir2,'all_clus_property.npy')

f1 = np.load(file_load1,allow_pickle=True)
f2 = np.load(file_load2,allow_pickle=True)

main_df1 = pd.DataFrame()
main_df2 = pd.DataFrame()

for iter in range(f1.size):
    tmp_df = pd.DataFrame(f1[iter])
    main_df1 = pd.concat([main_df1,tmp_df],axis = 0)
    
for iter in range(f2.size):
    tmp_df = pd.DataFrame(f1[iter])
    main_df2 = pd.concat([main_df2,tmp_df],axis = 0)

# fixing cluster IDs    
c_prev = main_df1['cluster_id'].max()    
main_df2['cluster_id'] = main_df2['cluster_id'] + c_prev

main_df = pd.concat([main_df1,main_df2],axis = 0)


# Plot for : Populational: neuron numbers of increased/lowered firing 
main_df_counts = main_df.filter(['cluster_id','plasticity_metric'])
main_df_counts = main_df_counts.drop_duplicates()

sns.countplot(x = 'plasticity_metric', data = main_df_counts,palette = ['r','#6F8FAF','g'])


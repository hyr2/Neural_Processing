#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 22:43:35 2023

@author: hyr2-office
"""

# Temporary Code to read JiaaoZ's .npz file outputs

import numpy as np
import os, sys
import pandas as pd
from matplotlib import pyplot as plt
import copy

# Creating the dataframe
dataframe_col = pd.DataFrame(
    {
    "day": [],
    "ms_connect": [],
    "region": []
    }
    )


# Greater 300 um shanks (within)
filename_input_G300 = '/home/hyr2-office/Documents/Data/NVC/temp/GT300/'
list_file = os.listdir(filename_input_G300)

# loading all data 
dict_all = {}
Series_day = pd.Series([],dtype = np.int8)
Series_ms = pd.Series([],dtype = float)
Series_region = pd.Series([],dtype = object)
for iter_n,file_n in enumerate(list_file):
    tmp_data = np.load(os.path.join(filename_input_G300,file_n))
    Series_day = pd.concat([Series_day,pd.Series(tmp_data['days'])])
    Series_ms = pd.concat([Series_ms,pd.Series(tmp_data['dataset'])])

Series_region = pd.Series(['G300']*len(Series_day),dtype = str)
df_GT300 = copy.deepcopy(dataframe_col)
df_GT300['day'] = Series_day
df_GT300['ms_connect'] = Series_ms
df_GT300['region'] = Series_region


# Less 300 um shanks (within)
filename_input_G300 = '/home/hyr2-office/Documents/Data/NVC/temp/LT300/'
list_file = os.listdir(filename_input_G300)

# loading all data 
dict_all = {}
Series_day = pd.Series([],dtype = np.int8)
Series_ms = pd.Series([],dtype = float)
Series_region = pd.Series([],dtype = object)
for iter_n,file_n in enumerate(list_file):
    tmp_data = np.load(os.path.join(filename_input_G300,file_n))
    Series_day = pd.concat([Series_day,pd.Series(tmp_data['days'])])
    Series_ms = pd.concat([Series_ms,pd.Series(tmp_data['dataset'])])

Series_region = pd.Series(['L300']*len(Series_day),dtype = str)
df_LT300 = copy.deepcopy(dataframe_col)
df_LT300['day'] = Series_day
df_LT300['ms_connect'] = Series_ms
df_LT300['region'] = Series_region

# S2 um shanks (within)
filename_input_G300 = '/home/hyr2-office/Documents/Data/NVC/temp/S2/'
list_file = os.listdir(filename_input_G300)

# loading all data 
dict_all = {}
Series_day = pd.Series([],dtype = np.int8)
Series_ms = pd.Series([],dtype = float)
Series_region = pd.Series([],dtype = object)
for iter_n,file_n in enumerate(list_file):
    tmp_data = np.load(os.path.join(filename_input_G300,file_n))
    Series_day = pd.concat([Series_day,pd.Series(tmp_data['days'])])
    Series_ms = pd.concat([Series_ms,pd.Series(tmp_data['dataset'])])

Series_region = pd.Series(['L300']*len(Series_day),dtype = str)
df_S2 = copy.deepcopy(dataframe_col)
df_S2['day'] = Series_day
df_S2['ms_connect'] = Series_ms
df_S2['region'] = Series_region


# plotting
f, ax1 = plt.subplots(1,1)
df_LT300.groupby('day').mean().plot(ax = ax1,linewidth = 2.9)
df_GT300.groupby('day').mean().plot(ax = ax1,linewidth = 2.9)
ax1.legend(['Peri-Infarct','Spared Barrels'])
ax1.set_ylabel('Within region Connectivity')
f.savefig(os.path.join('/home/hyr2-office/Documents/Paper/Single-Figures-SVG/Fig3/subfigures/monosyn_jiaaoz_fig3.svg'),format = 'svg')

# plotting
f, ax1 = plt.subplots(1,1)
df_LT300.groupby('day').mean().plot(ax = ax1,linewidth = 2.9)
df_S2.groupby('day').mean().plot(ax = ax1,linewidth = 2.9)
ax1.legend(['Peri-Infarct','wS2'])
ax1.set_ylabel('Within region Connectivity')
f.savefig(os.path.join('/home/hyr2-office/Documents/Paper/Single-Figures-SVG/Fig4/subfigures/monosyn_jiaaoz_fig4.svg'),format = 'svg')

# concat dataframes
df_final = pd.concat([df_GT300,df_LT300],axis = 0)



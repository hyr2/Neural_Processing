#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:58:28 2023

@author: hyr2-office
"""

# This contains functions that are used to merge multiple shanks into combined .npy files for each particular animal.
# THe function is called in the main script: rickshaw_postprocess_neuralHR.py

import scipy.io as sio # Import function to read data.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy
import seaborn as sns
import os, copy
from natsort import natsorted
import pandas as pd
import sys
import skfda
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)
import sklearn.cluster as skC
import sklearn.metrics as skM

def sort_cell_type(input_arr,shank_arr):
    # Function counts the number of wide, narrow and pyramidal cells from the matlab output (.mat file called pop_celltypes.mat)
    output_arr = np.zeros([3,4],dtype = np.int16)
    output_list_string = []
    if not input_arr.shape:
        return (output_arr,output_list_string)
    else:
        for iter in range(input_arr.shape[1]):
            str_celltype = input_arr[0][iter]
            if str_celltype == 'Pyramidal Cell':
                output_arr[0,shank_arr[iter]] += 1 
                output_list_string.append('P')
            elif str_celltype == 'Narrow Interneuron':
                output_arr[1,shank_arr[iter]] += 1 
                output_list_string.append('NI')
            elif str_celltype == 'Wide Interneuron':
                output_arr[2,shank_arr[iter]] += 1 
                output_list_string.append('WI')
        return (output_arr,output_list_string)

def combine_shanks(input_dir):
    source_dir_list = natsorted(os.listdir(input_dir))
    names_datasets = []
    pop_stats = {}
    clus_property = {}
    celltype = {}
    iter=0
    for name in source_dir_list:
        if os.path.isdir(os.path.join(input_dir,name)):
            folder_loc_mat = os.path.join(input_dir,name)
            if os.path.isdir(folder_loc_mat):
                pop_stats[iter] = np.load(os.path.join(folder_loc_mat,'Processed/count_analysis/all_clus_pca_preprocessed.npy'),allow_pickle=True)      # comes from population_analysis.py 
                clus_property[iter] = np.load(os.path.join(folder_loc_mat,'Processed/count_analysis/all_clus_property.npy'),allow_pickle=True)  # comes from population_analysis.py
                celltype[iter] = sio.loadmat(os.path.join(folder_loc_mat,'Processed/cell_type/pop_celltypes.mat'))  # comes from func_CE_BarrelCortex.m 
                names_datasets.append(name)
                iter += 1


    # PCA preprocessing raw data concatenated
    lst_1st = pop_stats[0]
    for iter_r in range(1,iter):
        lst_1st = np.concatenate((lst_1st,pop_stats[iter_r]),axis = 1)
    
    filename_save = os.path.join(input_dir,'all_shanks_pca_preprocessed.npy')
    np.save(filename_save,lst_1st)

    # cell types aggregated
    list_celltype_full = []
    for iter_l in range(len(celltype)):
        tmp_shank = celltype[iter_l]['shank_num']
        tmp_shank = np.squeeze(tmp_shank)
        tmp_shank = tmp_shank-1             # starts from 0 (consistent with python)
        (_,list_celltype) = sort_cell_type(celltype[iter_l]['celltype'],tmp_shank)
        list_celltype_full.extend(list_celltype)
    filename_save = os.path.join(input_dir,'all_shanks_celltype_processed.npy')
    np.save(filename_save,list_celltype_full)
    
    # plasticity metrics aggregated (based on Z score from population_analysis.py)
    # lst_plasticity_metric = []
    lst_clust_property = []
    for iter_l in range(len(clus_property)):
        for iter_i in range(clus_property[iter_l].size):
            lst_clust_property.append(clus_property[iter_l][iter_i])
            # lst_plasticity_metric.append(clus_property[iter_l][iter_i]['plasticity_metric'])
    
    filename_save = os.path.join(input_dir,'all_shanks_clus_property_processed.npy')
    np.save(filename_save,lst_clust_property)
    





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:15:37 2022

@author: hyr2-office
"""

import scipy.io as sio # Import function to read data.
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import os
from natsort import natsorted
import pandas as pd
import sys

# Plotting fonts
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=16)          # controls default text sizes

# Used to process one single animal, all sessions
# Make sure the codes: population_analysis.py is run as well as func_CE_BarrelCortex.m before this (see the rickshaw post processing file)

def sort_by_shank(type_local,shank_num_local):
    # This function is used to assign the excitatory and inhibitory type neurons to their respective shanks (A,B,C,D)
    # INPUTS: 
    #    type_local: array of boolean for excitatory/inhibitory neurons (1 -> excitatory_cell, 0 -> not excitatory_cell)
    #    shank_num_local: an array the same size as type_local containing the shank info of each cell/cluster 
    output_main = np.zeros([4,])
    for iter in range(4):
        output_local = np.logical_and(type_local,shank_num_local == iter)     
        output_main[iter] = np.sum(output_local)
        
    return output_main

def sort_single_shank_neuralAct_allclus(shank_num_mask,curation_mask_list,spikes_bsl,spikes_stim,input_df,shank_ID,session_ID,x_ticks_labels):
    # INPUTS:
    # shank_num_mask : a 1D array (length N) containing the shank label for each cluster [0 to 3]
    # curation_mask_list : a 1D boolean array (length N) containing curation label for each cluster
    # spikes_bsl : a 1D array (length N) containing the number of spikes pre-stimulation for each cluster
    # spikes_stim : a 1D array (length N) containing the number of spikes post-stimulation for each cluster
    # input_df : the input dataframe of type df_all_clusters (search code)
    # shank_ID : Data of this shank will be extracted
    
    output_df = pd.DataFrame(
        {
            "session" : [],
            "act" : [],
            "sup" : [],
            "E" : [],
            "I" : [],
            "stim_E" : [],
            "stim_I" : []
        }
    )
    
    temp_series1 = pd.Series([],dtype = 'float')
    temp_series2 = pd.Series([],dtype = 'float')
    temp_series3 = pd.Series([],dtype = 'float')
    temp_series4 = pd.Series([],dtype = 'float')
    temp_series5 = pd.Series([],dtype = 'float')
    temp_series6 = pd.Series([],dtype = 'float')
    
    tmp_bsl1 = spikes_bsl[curation_mask_list[0]]
    tmp_stim1 = spikes_stim[curation_mask_list[0]]   
    tmp_shank1 = shank_num_mask[curation_mask_list[0]]       # extracting the shank info of each cluster    
    tmp_bsl2 = spikes_bsl[curation_mask_list[1]]
    tmp_stim2 = spikes_stim[curation_mask_list[1]]
    tmp_shank2 = shank_num_mask[curation_mask_list[1]]
    tmp_bsl3 = spikes_bsl[curation_mask_list[2]]
    tmp_stim3 = spikes_stim[curation_mask_list[2]]  
    tmp_shank3 = shank_num_mask[curation_mask_list[2]]
    tmp_bsl4 = spikes_bsl[curation_mask_list[3]]
    tmp_stim4 = spikes_stim[curation_mask_list[3]]  
    tmp_shank4 = shank_num_mask[curation_mask_list[3]]
    tmp_bsl5 = spikes_bsl[curation_mask_list[4]]
    tmp_stim5 = spikes_stim[curation_mask_list[4]]  
    tmp_shank5 = shank_num_mask[curation_mask_list[4]]
    tmp_bsl6 = spikes_bsl[curation_mask_list[5]]
    tmp_stim6 = spikes_stim[curation_mask_list[5]]  
    tmp_shank6 = shank_num_mask[curation_mask_list[5]]
    
    iter_local = shank_ID
    
    tmp_bsl_local = tmp_bsl1[tmp_shank1 == iter_local]
    tmp_stim_local = tmp_stim1[tmp_shank1 == iter_local]
    if tmp_bsl_local.size != 0:                                               # check if empty array
        temp_series1 = pd.Series(np.squeeze( tmp_stim_local - tmp_bsl_local  ))
    tmp_bsl_local = tmp_bsl2[tmp_shank2 == iter_local]
    tmp_stim_local = tmp_stim2[tmp_shank2 == iter_local]
    if tmp_bsl_local.size != 0:                                               # check if empty array
        temp_series2 = pd.Series(np.squeeze( tmp_stim_local - tmp_bsl_local)  )
    tmp_bsl_local = tmp_bsl3[tmp_shank3 == iter_local]
    tmp_stim_local = tmp_stim3[tmp_shank3 == iter_local]
    if tmp_bsl_local.size != 0:                                               # check if empty array
        temp_series3 = pd.Series(np.squeeze( tmp_stim_local - tmp_bsl_local)  )
    tmp_bsl_local = tmp_bsl4[tmp_shank4 == iter_local]
    tmp_stim_local = tmp_stim4[tmp_shank4 == iter_local]
    if tmp_bsl_local.size != 0:                                               # check if empty array
        temp_series4 = pd.Series(np.squeeze( tmp_stim_local - tmp_bsl_local)  )
    tmp_bsl_local = tmp_bsl5[tmp_shank5 == iter_local]
    tmp_stim_local = tmp_stim5[tmp_shank5 == iter_local]
    if tmp_bsl_local.size != 0:                                               # check if empty array
        temp_series5 = pd.Series(np.squeeze( tmp_stim_local - tmp_bsl_local)  )
    tmp_bsl_local = tmp_bsl6[tmp_shank6 == iter_local]
    tmp_stim_local = tmp_stim6[tmp_shank6 == iter_local]
    if tmp_bsl_local.size != 0:                                               # check if empty array
        temp_series6 = pd.Series(np.squeeze( tmp_stim_local - tmp_bsl_local)  )

    output_df.act = temp_series2
    output_df.sup = temp_series1
    output_df.E = temp_series3
    output_df.I = temp_series4
    output_df.stim_E = temp_series5
    output_df.stim_I = temp_series6
    output_df.session = pd.Series( x_ticks_labels[session_ID] * np.ones([len(output_df),],dtype = np.int16),dtype = np.int16)
    
    output_df.reset_index(drop =True, inplace = True)
    
    output_df = pd.concat([output_df,input_df],ignore_index=True)
    
    output_df.reset_index(drop = True, inplace = True)
    
    return output_df

def sort_by_shank_neuralAct(shank_num_mask,curation_mask,spikes_bsl,spikes_stim,spont_FR,event_FR):
    # INPUTS:
    # shank_num_mask : a 1D array (length N) containing the shank label for each cluster [0 to 3]
    # curation_mask : a 1D boolean array (length N) containing curation label for each cluster
    # spikes_bsl : a 1D array (length N) containing the number of spikes pre-stimulation for each cluster
    # spikes_stim : a 1D array (length N) containing the number of spikes post-stimulation for each cluster
    tmp_bsl = spikes_bsl[curation_mask]
    tmp_stim = spikes_stim[curation_mask]
    tmp_spont = spont_FR[curation_mask]
    tmp_event = event_FR[curation_mask]
    tmp_shank = shank_num_mask[curation_mask]
    activity_nor = np.zeros([4,],dtype=float)
    activity_non = np.zeros([4,],dtype=float)
    activity_non_abs = np.zeros([4,],dtype=float)
    activity_spont = np.zeros([4,],dtype=float)     # avg baseline FR non-stimulus
    activity_event = np.zeros([4,],dtype=float)     # peak FR during stimulus 
    for iter_local in range(4):   # 4 shanks
        tmp_bsl_local = tmp_bsl[tmp_shank == iter_local]
        tmp_stim_local = tmp_stim[tmp_shank == iter_local]
        tmp_spont_local = tmp_spont[tmp_shank == iter_local]
        tmp_event_local = tmp_event[tmp_shank == iter_local]
        if tmp_bsl_local.size != 0:                                               # check if empty array
            activity_nor[iter_local] = np.mean(tmp_stim_local/tmp_bsl_local - 1)    # Normlaized neural activty 
            activity_non[iter_local] = np.mean(tmp_stim_local - tmp_bsl_local)      # Non-normlaized neural activty (subtracting the baseline activity)
            activity_non_abs[iter_local] = np.mean(tmp_stim_local)                  # Non-normalized neural activity (only the spikes during the stim duration)
            activity_spont[iter_local] = np.mean(tmp_spont_local)                   # avg baseline FR non-stimulus
            activity_event[iter_local] = np.nanmean(tmp_event_local)             # peak FR during stimulus 
    return (activity_nor, activity_non, activity_non_abs,activity_spont,activity_event)
    
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
    
def sort_cell_type_2(shank_num_mask,curation_mask):
    # Function sorts (into shanks) the number of Activated and Suppressed E and I cells from the matlab output
    output_arr = np.zeros([4,],dtype = np.int32)
    tmp_shank = shank_num_mask[curation_mask]
    
    for iter_local in range(4):
        sst_junk = tmp_shank[tmp_shank == iter_local]
        if sst_junk.size != 0:
            output_arr[iter_local] = sst_junk.size
        else:
            output_arr[iter_local] = 0
        
    return output_arr
    
def convert2df(T2P_allsessions):
    # Function is being used to organize the array for trough to peak time (ms) 
    # the input is a dictionary of multiple numpy arrays of different lengths. 
    # Each index of the dictionary represents one session
    # The output is a dataframe to be used for better data organization and plotting
    df_excit = pd.DataFrame(
            {
                "index": [],
                "T2P": []
            }
        )
    df_inhib = pd.DataFrame(
            {
                "index": [],
                "T2P": []
            }
        )
    T2P = pd.Series([],dtype = 'float')
    for iter_local in range(len(T2P_allsessions)):
        T2P = pd.concat([T2P,pd.Series(np.squeeze(T2P_allsessions[iter_local]))])
    
    tmp_bool = np.array(T2P > 0.47)
    df_excit.T2P = T2P[tmp_bool]
    df_excit.index = np.linspace(0,df_excit.T2P.shape[0]-1,df_excit.T2P.shape[0])
    tmp_bool = np.array(T2P <= 0.47)
    df_inhib.T2P = T2P[tmp_bool]
    df_inhib.index = np.linspace(0,df_inhib.T2P.shape[0]-1,df_inhib.T2P.shape[0])
    return df_excit,df_inhib

def extract_spikes(clus_property_local):
    N_stim = np.zeros([len(clus_property_local),],dtype = float)
    N_bsl = np.zeros([len(clus_property_local),],dtype = float)
    cluster_propery = np.zeros([len(clus_property_local),],dtype = np.int8)
    shank_num = np.ones([len(clus_property_local),],dtype = np.int8)
    spont_FR = np.zeros([len(clus_property_local),],dtype = float)
    event_FR = np.zeros([len(clus_property_local),],dtype = float)
    for itr in range(len(clus_property_local)):
        N_stim[itr] = clus_property_local[itr]['N_spikes_stim']     # Number of spikes during stimulation(Trial averaged)
        N_bsl[itr] = clus_property_local[itr]['N_spikes_bsl']       # Number of spikes pre-stimulation(Trial averaged)
        cluster_propery[itr] = clus_property_local[itr]['clus_prop']
        shank_num[itr] = clus_property_local[itr]['shank_num']       # starts from 0
        spont_FR[itr] = clus_property_local[itr]['spont_FR']    # in Hz (Trial averaged)
        event_FR[itr] = clus_property_local[itr]['EventRelatedFR']    # in Hz (Trial averaged)
    return (cluster_propery,N_stim,N_bsl,shank_num,spont_FR,event_FR)
    
def extract_bursts(clus_property_local):
    n_clusters = len(clus_property_local)
    burstN = np.zeros([n_clusters,],dtype = float)
    burstN_bsl = np.zeros([n_clusters,],dtype = float)
    burstL = np.zeros([n_clusters,],dtype = float)
    burstL_bsl = np.zeros([n_clusters,],dtype = float)
    burstFR = np.zeros([n_clusters,],dtype = float)
    burstFR_bsl = np.zeros([n_clusters,],dtype = float)
    for itr in range(n_clusters):
        burstN[itr] = clus_property_local[itr]['burstN']
        burstN_bsl[itr] = clus_property_local[itr]['burstN_bsl']
        burstL[itr] = clus_property_local[itr]['burstL']
        burstL_bsl[itr] = clus_property_local[itr]['burstL_bsl']
        burstFR[itr] = clus_property_local[itr]['burstFR']
        burstFR_bsl[itr] = clus_property_local[itr]['burstFR_bsl']
    return (burstN,burstN_bsl,burstL,burstL_bsl,burstFR,burstFR_bsl)    

def extract_adaptation(clus_property_local):
    n_clusters = len(clus_property_local)
    adapt_time_avg = np.zeros([n_clusters,],dtype = float)  
    adapt_time_end = np.zeros([n_clusters,],dtype = float)
    adapt_trial = np.zeros([n_clusters,],dtype = float)
    for itr in range(n_clusters):
        adapt_time_avg[itr] = clus_property_local[itr]['adapt_time_avg']
        adapt_time_end[itr] = clus_property_local[itr]['adapt_time_end']
        adapt_trial[itr] = clus_property_local[itr]['adapt_trial']
    return (adapt_time_avg,adapt_time_end,adapt_trial)    

def combine_sessions(source_dir, str_ID):
    # source_dir : source directory global (ie for mouse instead of a single session)
    # str_ID : string for mouse label (name/ID)
    # source_dir = '/home/hyr2-office/Documents/Data/NVC/RH-8/'
    # rmv_bsl = input('Baselines to remove (specify as index: e.g: 0, 2)? Select -1 for no baselines.\n')             # specify what baseline datasets need to be removed from the analysis
    source_dir_list = natsorted(os.listdir(source_dir))
    # Preparing variables
    # rmv_bsl = rmv_bsl.split(',')
    # rmv_bsl = np.asarray(rmv_bsl, dtype = np.int8)
    # if not np.any(rmv_bsl == -1):
        # source_dir_list = np.delete(source_dir_list,rmv_bsl)
        # source_dir_list = source_dir_list.tolist()
    # source_dir_list = source_dir_list.tolist()
    # source_dir_list = natsorted(os.listdir(source_dir))

    if (str_ID.lower() == 'processed_data_rh3'.lower()):
        linear_xaxis = np.array([-3,-2,2,7,14,21,28,56]) 
        dict_shank_spatial_info = {
            '0':'NaN',
            '1':'L300',
            '2':'G300',
            '3':'NaN'
        }
    elif (str_ID.lower() == 'processed_data_bc7'.lower()):
        linear_xaxis = np.array([-3,-2,2,7,14,21,28,42]) 
        dict_shank_spatial_info = {
            '0':'L300',
            '1':'G300',
            '2':'S2',
            '3':'S2'
        }
    elif (str_ID.lower() == 'BC6'.lower()):
        linear_xaxis = np.array([-3,-2,2,9,14,21,28,35,49])
    elif (str_ID.lower() == 'processed_data_bbc5'.lower()):
        linear_xaxis = np.array([-3,-2,2,7,14,21,47]) 
        dict_shank_spatial_info = {
            '0':'NaN',
            '1':'NaN',
            '2':'L300',
            '3':'NaN'
        }
    elif (str_ID.lower() == 'processed_data_rh7'.lower()):
        linear_xaxis = np.array([-3,-2,2,7,14,24,28,35,42,49,56])
        dict_shank_spatial_info = {
            '0':'G300',
            '1':'L300',
            '2':'L300',
            '3':'S2'
        }
    elif (str_ID.lower() == 'BC8'.lower()):
        linear_xaxis = np.array([-3,-2,2,2,7,8,15,21,54])
    elif (str_ID.lower() == 'processed_data_rh8'.lower()):
        linear_xaxis = np.array([-3,-2,2,7,14,21,28,35,42,49,56])  
        dict_shank_spatial_info = {
            '0':'L300',
            '1':'L300',
            '2':'NaN',
            '3':'S2'
        }
    elif (str_ID.lower() == 'processed_data_rh9'.lower()):
        linear_xaxis = np.array([-3,-2,2,7,14,21,28,35,42,49])
        dict_shank_spatial_info = {
            '0':'L300',
            '1':'L300',
            '2':'G300',
            '3':'NaN'
        }
    elif (str_ID.lower() == 'B-BC8'.lower()):
        linear_xaxis = np.array([-4,-3,-2,-1,3,7])
    elif (str_ID.lower() == 'BHC-7'.lower()):
        linear_xaxis = np.array([-3,-2,-1,7,14])
    elif (str_ID.lower() == 'processed_data_rh11'.lower()):
        linear_xaxis = np.array([-3,-2,2,7,14,21,28,35,42])
        dict_shank_spatial_info = {
            '0':'G300',
            '1':'L300',
            '2':'L300',
            '3':'G300'
        }
    else:
        sys.exit('No string matched with: ' + str_ID)
            
           
    # x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14','Day 21','Day 28','Day 56']  # RH3 (reject baselines 0 and 2)
    # linear_xaxis = np.array([-2,-1,2,7,14,21,28,56]) 
    # x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14 ','Day 21','Day 28','Day 42'] # BC7 (reject baseline 0)
    # linear_xaxis = np.array([-2,-1,2,7,14,21,28,42]) 
    # x_ticks_labels = ['bl-1','bl-2','Day 2','Day 9','Day 14 ','Day 21','Day 28','Day 35','Day 49'] # BC6 (stroke not formed at all. Data should be rejected)
    # linear_xaxis = np.array([-2,-1,2,9,14,21,28,35,49]) 
    # x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14','Day 21','Day 47'] # B-BC5
    # linear_xaxis = np.array([-2,-1,2,7,14,21,47]) 
    # x_ticks_labels = ['bl-1','bl-2','bl-3','Day 2','Day 7','Day 14 ','Day 24','Day 28','Day 35','Day 42','Day 49','Day 56'] # R-H7 (main)
    # linear_xaxis = np.array([-3,-2,-1,2,7,14,24,28,35,42,49,56]) # 24 special for rh7
    # x_ticks_labels = ['bl-1','Day 2','Day 7','Day 14 ','Day 21','Day 42'] # BC8 
    # linear_xaxis = np.array([-3,-2,-1,2,2,7,8,14,21,54])            
    # x_ticks_labels = ['bl-1','bl-2','Day 2','Day 7','Day 14 ','Day 21','Day 42'] # R-H8 
    # linear_xaxis = np.array([-2,-1,2,7,14,21,28,35,42,49])            
    # linear_xaxis = np.array([-3,-2,-1,2,7,14,21,28,35,42,49])  # RH-9

    x_ticks_labels = linear_xaxis

    pop_stats = {}
    pop_stats_cell = {}
    clus_property = {}
    names_datasets = []
    iter = 0
    # Loading all longitudinal data into dictionaries 
    for name in source_dir_list:
        if os.path.isdir(os.path.join(source_dir,name)):
            folder_loc_mat = os.path.join(source_dir,name)
            if os.path.isdir(folder_loc_mat):
                pop_stats[iter] = sio.loadmat(os.path.join(folder_loc_mat,'Processed/count_analysis/population_stat_responsive_only.mat'))      # comes from population_analysis.py 
                clus_property[iter] = np.load(os.path.join(folder_loc_mat,'Processed/count_analysis/all_clus_property.npy'),allow_pickle=True)  # comes from population_analysis.py
                pop_stats_cell[iter] = sio.loadmat(os.path.join(folder_loc_mat,'Processed/cell_type/pop_celltypes.mat'))                        # comes from cell explorer MATLAB
                names_datasets.append(name)
                iter += 1
            
    act_nclus = np.zeros([len(pop_stats),4])
    act_E_nclus = np.zeros([len(pop_stats),4])
    act_I_nclus = np.zeros([len(pop_stats),4])
    act_FR = np.zeros([len(pop_stats),4])
    suppressed_nclus = np.zeros([len(pop_stats),4])
    inh_FR = np.zeros([len(pop_stats),4])
    nor_nclus = np.zeros([len(pop_stats),4])
    N_chan_shank = np.zeros([len(pop_stats),4])
    act_nclus_total = np.zeros([len(pop_stats),])
    suppressed_nclus_total = np.zeros([len(pop_stats),])
    celltype_total = np.zeros([len(pop_stats),3])
    celltype_shank = np.zeros([len(pop_stats),3,4])
    excitatory_cell = np.zeros([len(pop_stats),4]) # by shank
    inhibitory_cell = np.zeros([len(pop_stats),4]) # by shank
    activity_nor = np.zeros([len(pop_stats),6,4])    # lots of FR by cluster/response type  Normalized to baseline
    activity_non = np.zeros([len(pop_stats),6,4])    # lots of FR by cluster/response type  Non normalized
    activity_spont = np.zeros([len(pop_stats),6,4])    # lots of FR by cluster/response type  Non normalized (spontaneous ie outside of the stimulation period)
    activity_event = np.zeros([len(pop_stats),6,4])    # lots of FR by cluster/response type  Non normalized (spontaneous ie outside of the stimulation period)
    activity_non_abs = np.zeros([len(pop_stats),6,4])  # lots of FR by cluster/response type  Non normalized. Count of number of spikes during stim
    clus_N = np.zeros([len(pop_stats),6])
    df_all_clusters = pd.DataFrame(
        {
            "session": [],
            "act" : [],
            "sup" : [],
            "E" : [],
            "I" : [],
            "stim_E" : [],
            "stim_I" : []
        }
    )
    df_all_clusters_main = pd.DataFrame(
        {
                "animal_ID":[],
                "day": [],
                "shank": [],
                "celltype": [],
                "response": [],
                "T2P": [],
                "spont_FR": [], # the spont FR pre stim 
                "event_FR": [], # Peak FR during stim (for inhibited cells, this would be the minimum FR)
                "N_bsl" : [],    # Total spikes in 1.5 sec duration
                "N_stim" : [],  # Total spikes in 1.5 sec duration
                "burstN" : [],     #during stimulation only (average over trials) [exact times: 2.5s to 4s mark]
                "burstL" : [],     #during stimulation only (length averaged over all events irrespective of trials) [exact times: 2.5s to 4s mark]
                "burstFR": [],     #during stimulation only (FR averaged over all events irrespective of trials) [exact times: 2.5s to 4s mark]
                "burstN_bsl" : [],     #during baseline only (average over trials) [exact times: 0.5 to 2.5s mark]
                "burstL_bsl" : [],     #during baseline only (length averaged over all events irrespective of trials) [exact times: 0.5 to 2.5s mark]
                "burstFR_bsl": [],     #during baseline only (FR averaged over all events irrespective of trials) [exact times: 0.5 to 2.5s  mark]
                "burst_i": [],
                "tau_r": [],
                "wav_assym": [],
                "adapt_time_avg" : [],  # Thomas Adaptation Z scored relavtive to avg response during stim
                "adapt_time_end" : [],  # Thomas Adaptation Z scored relavtive to end of response during stim
                "adapt_trial" : [],      # Thomas Adaptation over trials
                "region" : []
        }
    )
    
    df_all_clusters_A = deepcopy(df_all_clusters)
    df_all_clusters_B = deepcopy(df_all_clusters)
    df_all_clusters_C = deepcopy(df_all_clusters)
    df_all_clusters_D = deepcopy(df_all_clusters)
    # FR_new = np.zeros([len(pop_stats),3,3])     
    # celltype_excit = np.zeros([len(pop_stats),3])
    # celltype_inhib = np.zeros([len(pop_stats),3])
    T2P_allsessions = []    # list of 1D numpy arrays
    main_df = df_all_clusters_main.copy()
    avg_spont_FR = []
    avg_stim_FR = []
    for iter in range(len(pop_stats)):          # loop over sessions of a single animal
        # population extraction from dictionaries
        
        # FR goes up (activated neurons)
        act_nclus[iter,0] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[0]     # Shank A
        act_nclus[iter,1] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[1]     # Shank B 
        act_nclus[iter,2] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[2]     # Shank C
        act_nclus[iter,3] = np.squeeze(pop_stats[iter]['act_nclus_by_shank'])[3]     # Shank D
        act_FR[iter,0] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[0]
        act_FR[iter,1] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[1]
        act_FR[iter,2] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[2]
        act_FR[iter,3] = np.squeeze(pop_stats[iter]['avg_FR_act_by_shank'])[3]
        # FR goes down (suppressed activity)
        suppressed_nclus[iter,0] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[0]     # Shank A
        suppressed_nclus[iter,1] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[1]     # Shank B 
        suppressed_nclus[iter,2] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[2]     # Shank C
        suppressed_nclus[iter,3] = np.squeeze(pop_stats[iter]['inh_nclus_by_shank'])[3]     # Shank C
        inh_FR[iter,0] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[0]
        inh_FR[iter,1] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[1]
        inh_FR[iter,2] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[2]
        inh_FR[iter,3] = np.squeeze(pop_stats[iter]['avg_FR_inh_by_shank'])[3]
        # No response clusters
        nor_nclus[iter,0] = np.squeeze(pop_stats[iter]['nor_nclus_by_shank'])[0] 
        nor_nclus[iter,1] = np.squeeze(pop_stats[iter]['nor_nclus_by_shank'])[1] 
        nor_nclus[iter,2] = np.squeeze(pop_stats[iter]['nor_nclus_by_shank'])[2] 
        nor_nclus[iter,3] = np.squeeze(pop_stats[iter]['nor_nclus_by_shank'])[3] 
        # Number of channels per shank in each session
        N_chan_shank[iter,0] = np.squeeze(pop_stats[iter]['numChan_perShank'])[0]
        N_chan_shank[iter,1] = np.squeeze(pop_stats[iter]['numChan_perShank'])[1]
        N_chan_shank[iter,2] = np.squeeze(pop_stats[iter]['numChan_perShank'])[2]
        N_chan_shank[iter,3] = np.squeeze(pop_stats[iter]['numChan_perShank'])[3]
        
        act_nclus_total[iter] = np.sum(act_nclus[iter,:])
        suppressed_nclus_total[iter] = np.sum(suppressed_nclus[iter,:])
        
        # cell type extraction from dictionaries
        tmp = pop_stats_cell[iter]['celltype']
        tmp_shank = pop_stats_cell[iter]['shank_num']
        tmp_shank = np.squeeze(tmp_shank)
        tmp_shank = tmp_shank-1             # starts from 0 (consistent with python)
        (celltype_shank[iter,:],list_celltype) = sort_cell_type(tmp,tmp_shank)
        celltype_total[iter,:] = np.sum(celltype_shank[iter,:],axis = 1)
        # excitatory and inhibitory neuron populations
        excitatory_cell[iter,:] = celltype_shank[iter,0,:]
        inhibitory_cell[iter,:] = celltype_shank[iter,1,:]
        # excitatory_cell[iter,:] = sort_by_shank(pop_stats_cell[iter]['type_excit'],tmp_shank)
        # inhibitory_cell[iter,:] = sort_by_shank(pop_stats_cell[iter]['type_inhib'],tmp_shank)
        # Saving spike counts
        (cluster_property,N_stim,N_bsl,shank_num,spont_FR,event_FR) = extract_spikes(clus_property[iter])
        (burstN,burstN_bsl,burstL,burstL_bsl,burstFR,burstFR_bsl) = extract_bursts(clus_property[iter])
        (adapt_time_avg,adapt_time_end,adapt_trial) = extract_adaptation(clus_property[iter])
        # Avg Spont FR (Tonic activity) 
        avg_spont_FR.append(np.mean(spont_FR))
        # Avg during stim FR (phasic activity)
        avg_stim_FR.append(np.nanmean(event_FR))
        
        # Spike analysis (neural activity) [each animal has an equal footing ie data is prepared for averaging over animals]
        (activity_nor_local,activity_non_local,activity_non_abs_local,activity_spont_local,activity_event_local) = sort_by_shank_neuralAct(shank_num,np.squeeze(cluster_property == -1),N_bsl,N_stim,spont_FR,event_FR)
        activity_nor[iter,0,:] = activity_nor_local   # Normlaized neural activty (suppressed)
        activity_non[iter,0,:] = activity_non_local   # Non-normlaized neural activty suppressed
        activity_non_abs[iter,0,:] = activity_non_abs_local # number of spikes fired during the stimulation period by the suppressed neuron
        activity_spont[iter,0,:] = activity_spont_local     # spontaneous FR outside of stimulation
        activity_event[iter,0,:] = activity_event_local     # event related peak firing rate (averaged over clusters of course)
        (activity_nor_local,activity_non_local,activity_non_abs_local,activity_spont_local,activity_event_local) = sort_by_shank_neuralAct(shank_num,np.squeeze(cluster_property == 1),N_bsl,N_stim,spont_FR,event_FR)
        activity_nor[iter,1,:] = activity_nor_local   # Normlaized neural activty (activated)
        activity_non[iter,1,:] = activity_non_local   # Non-normlaized neural activty activated
        activity_non_abs[iter,1,:] = activity_non_abs_local # number of spikes fired during the stimulation period by the suppressed neuron
        activity_spont[iter,1,:] = activity_spont_local     # spontaneous FR outside of stimulation
        activity_event[iter,1,:] = activity_event_local     # event related peak firing rate (averaged over clusters of course)
        # FR by E and I
        mask_local = np.squeeze(pop_stats_cell[iter]['type_excit']) == 1
        (activity_nor_local,activity_non_local,activity_non_abs_local,activity_spont_local,activity_event_local) = sort_by_shank_neuralAct(shank_num,mask_local,N_bsl,N_stim,spont_FR,event_FR)
        activity_nor[iter,2,:] = activity_nor_local   # Normlaized neural activty (E cells)
        activity_non[iter,2,:] = activity_non_local   # Non-normlaized neural activty E cells
        activity_non_abs[iter,2,:] = activity_non_abs_local # number of spikes fired during the stimulation period by the suppressed neuron
        activity_spont[iter,2,:] = activity_spont_local     # spontaneous FR outside of stimulation
        activity_event[iter,2,:] = activity_event_local     # event related peak firing rate (averaged over clusters of course)
        mask_local = np.squeeze(pop_stats_cell[iter]['type_inhib']) == 1
        (activity_nor_local,activity_non_local,activity_non_abs_local,activity_spont_local,activity_event_local) = sort_by_shank_neuralAct(shank_num,mask_local,N_bsl,N_stim,spont_FR,event_FR)
        activity_nor[iter,3,:] = activity_nor_local   # Normlaized neural activty (I cells)
        activity_non[iter,3,:] = activity_non_local   # Non-normlaized neural activty I cells
        activity_non_abs[iter,3,:] = activity_non_abs_local # number of spikes fired during the stimulation period by the suppressed neuron
        activity_spont[iter,3,:] = activity_spont_local     # spontaneous FR outside of stimulation
        activity_event[iter,3,:] = activity_event_local     # event related peak firing rate (averaged over clusters of course)
        # FR of stimulus locked E cells 
        mask_local = np.logical_and(np.squeeze(pop_stats_cell[iter]['type_excit']) == 1, cluster_property == 1)
        (activity_nor_local,activity_non_local,activity_non_abs_local,activity_spont_local,activity_event_local) = sort_by_shank_neuralAct(shank_num,mask_local,N_bsl,N_stim,spont_FR,event_FR)
        activity_nor[iter,4,:] = activity_nor_local   # Normlaized neural activty (I cells)
        activity_non[iter,4,:] = activity_non_local   # Non-normlaized neural activty I cells
        activity_non_abs[iter,4,:] = activity_non_abs_local # number of spikes fired during the stimulation period by the suppressed neuron
        activity_spont[iter,4,:] = activity_spont_local     # spontaneous FR outside of stimulation
        activity_event[iter,4,:] = activity_event_local     # event related peak firing rate (averaged over clusters of course)
        # FR of stimulus locked I cells 
        mask_local = np.logical_and(np.squeeze(pop_stats_cell[iter]['type_inhib']) == 1, cluster_property == 1)
        (activity_nor_local,activity_non_local,activity_non_abs_local,activity_spont_local,activity_event_local) = sort_by_shank_neuralAct(shank_num,mask_local,N_bsl,N_stim,spont_FR,event_FR)
        activity_nor[iter,5,:] = activity_nor_local   # Normlaized neural activty (I cells)
        activity_non[iter,5,:] = activity_non_local   # Non-normlaized neural activty I cells
        activity_non_abs[iter,5,:] = activity_non_abs_local # number of spikes fired during the stimulation period by the suppressed neuron
        activity_spont[iter,5,:] = activity_spont_local     # spontaneous FR outside of stimulation
        activity_event[iter,5,:] = activity_event_local     # event related peak firing rate (averaged over clusters of course)
        # Spike analysis (neural activity) [each cluster has an equal footing ie data is prepared for averaging over clusters irrespective of animals]
        mask_local_list = [None for _ in range(6)]
        mask_local_list[0] =  np.squeeze(cluster_property == -1)    # suppressed
        mask_local_list[1] = np.squeeze(cluster_property == 1)      # activated
        mask_local_list[2] = np.squeeze(pop_stats_cell[iter]['type_excit']) == 1    # excitatory cells
        mask_local_list[3] = np.squeeze(pop_stats_cell[iter]['type_inhib']) == 1    # inhibitory cells
        mask_local_list[4] = np.logical_and(np.squeeze(pop_stats_cell[iter]['type_excit']) == 1, cluster_property == 1)
        mask_local_list[5] = np.logical_and(np.squeeze(pop_stats_cell[iter]['type_inhib']) == 1, cluster_property == 1)
        
        df_all_clusters_A = sort_single_shank_neuralAct_allclus(shank_num, mask_local_list, N_bsl, N_stim, df_all_clusters_A,shank_ID = 0,session_ID = iter,x_ticks_labels = x_ticks_labels)
        df_all_clusters_B = sort_single_shank_neuralAct_allclus(shank_num, mask_local_list, N_bsl, N_stim, df_all_clusters_B,shank_ID = 1,session_ID = iter,x_ticks_labels = x_ticks_labels)
        df_all_clusters_C = sort_single_shank_neuralAct_allclus(shank_num, mask_local_list, N_bsl, N_stim, df_all_clusters_C,shank_ID = 2,session_ID = iter,x_ticks_labels = x_ticks_labels)
        df_all_clusters_D = sort_single_shank_neuralAct_allclus(shank_num, mask_local_list, N_bsl, N_stim, df_all_clusters_D,shank_ID = 3,session_ID = iter,x_ticks_labels = x_ticks_labels)
        
        
        # number of clusters (total irrespective of the shank ID)
        clus_N[iter,0] = (cluster_property == -1).sum()     # number of suppressed clusters
        clus_N[iter,1] = (cluster_property == 1).sum()      # number of activated clusters
        clus_N[iter,2] = (np.squeeze(pop_stats_cell[iter]['type_excit']) == 1).sum()
        clus_N[iter,3] = (np.squeeze(pop_stats_cell[iter]['type_inhib']) == 1).sum()
        clus_N[iter,4] = (np.logical_and( np.squeeze(pop_stats_cell[iter]['type_excit']) == 1, cluster_property == 1)).sum()
        clus_N[iter,5] = (np.logical_and( np.squeeze(pop_stats_cell[iter]['type_inhib']) == 1, cluster_property == 1)).sum()
        mask_local = np.logical_and( np.squeeze(pop_stats_cell[iter]['type_excit']) == 1, cluster_property == 1)
        act_E_nclus[iter,:] = sort_cell_type_2(shank_num, mask_local)
        mask_local = np.logical_and( np.squeeze(pop_stats_cell[iter]['type_inhib']) == 1, cluster_property == 1)
        act_I_nclus[iter,:] = sort_cell_type_2(shank_num, mask_local)
        # Saving T2P for global histogram
        str_local = 'session_' + str(iter)
        T2P_allsessions.append(np.squeeze(pop_stats_cell[iter]['troughToPeak']))
        T2P_arr = np.squeeze(pop_stats_cell[iter]['troughToPeak'])  # trough to peak width
        wv_asym_arr = np.squeeze(pop_stats_cell[iter]['assymetry'])                 # The AB ratio (waveform assymetry following Destexhe's paper: https://www.sciencedirect.com/science/article/pii/S0969996118307605)
        tau_refractory_arr = np.squeeze(pop_stats_cell[iter]['tau_rise'])           # tau rise time (the fit of the triple exponential by Cell Explorer. This is the ACG refractory period of the cluster)
        burst_I_arr = np.squeeze(pop_stats_cell[iter]['burstIndex_Royer2012'])      # burst index (following Buzsaki 2017 paper definition: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5293146/pdf/nihms836070.pdf)
        # Dataframe for all clusters
        # tmp_df = df_all_clusters_main
        for iter_local in range(shank_num.shape[0]):
            print(shank_num[iter_local])
        
        # Get region here (from dictionary mapping see top of code)
        shank_num_list = list(map(str, list(shank_num)))
        shank_num_list = [dict_shank_spatial_info[key] for key in shank_num_list if key in dict_shank_spatial_info]

        tmp_df = pd.DataFrame(
            {
                "animal_ID":[str_ID] * T2P_arr.shape[0],
                "day": linear_xaxis[iter] * np.ones(T2P_arr.shape[0],dtype = np.int16),
                "shank": shank_num,
                "celltype": list_celltype,
                "response": cluster_property,
                "T2P": T2P_arr,
                "spont_FR": spont_FR, # the spont FR pre stim 
                "event_FR": event_FR, # Peak FR during stim (for inhibited cells, this would be the minimum FR)
                "N_bsl" : N_bsl,    # Total spikes in 1.5 sec duration
                "N_stim" : N_stim,  # Total spikes in 1.5 sec duration
                "burstN" : burstN,     #during stimulation only (average over trials) [exact times: 2.5s to 4s mark]
                "burstL" : burstL,     #during stimulation only (length averaged over all events irrespective of trials) [exact times: 2.5s to 4s mark]
                "burstFR": burstFR,     #during stimulation only (FR averaged over all events irrespective of trials) [exact times: 2.5s to 4s mark]
                "burstN_bsl" : burstN_bsl,     #during baseline only (average over trials) [exact times: 0.5 to 2.5s mark]
                "burstL_bsl" : burstL_bsl,     #during baseline only (length averaged over all events irrespective of trials) [exact times: 0.5 to 2.5s mark]
                "burstFR_bsl": burstFR_bsl,     #during baseline only (FR averaged over all events irrespective of trials) [exact times: 0.5 to 2.5s  mark]
                "burst_i": burst_I_arr,
                "tau_r": tau_refractory_arr,
                "wav_assym": wv_asym_arr,
                "adapt_time_avg" : adapt_time_avg,  # Thomas Adaptation Z scored relavtive to avg response during stim
                "adapt_time_end" : adapt_time_end,  # Thomas Adaptation Z scored relavtive to end of response during stim
                "adapt_trial" : adapt_trial,      # Thomas Adaptation over trials
                "region" : shank_num_list        # Region that this shank belongs to
            }
        )
        
        main_df = pd.concat([main_df,tmp_df],axis = 0)
        
        
        
    # combining     
    all_clusters_A = np.squeeze(df_all_clusters_A.to_numpy(dtype = float))
    all_clusters_B = np.squeeze(df_all_clusters_B.to_numpy(dtype = float))
    all_clusters_C = np.squeeze(df_all_clusters_C.to_numpy(dtype = float))
    all_clusters_D = np.squeeze(df_all_clusters_D.to_numpy(dtype = float))
    
    max_clus_in_shank = np.max(np.array([all_clusters_A.shape[0],all_clusters_B.shape[0],all_clusters_C.shape[0],all_clusters_D.shape[0]]))
    
    for iter in range(4): # loop over shanks
        if iter == 0:
            tmp_append = np.empty([max_clus_in_shank - all_clusters_A.shape[0],7],dtype = float)
            tmp_append[:] = np.nan
            if tmp_append.shape[0] != 0:
                all_clusters_A = deepcopy(np.append(all_clusters_A,tmp_append,axis = 0))
        elif iter == 1:
            tmp_append = np.empty([max_clus_in_shank - all_clusters_B.shape[0],7],dtype = float)
            tmp_append[:] = np.nan
            if tmp_append.shape[0] != 0:
                all_clusters_B = deepcopy(np.append(all_clusters_B,tmp_append,axis = 0))
        elif iter == 2:
            tmp_append = np.empty([max_clus_in_shank - all_clusters_C.shape[0],7],dtype = float)
            tmp_append[:] = np.nan
            if tmp_append.shape[0] != 0:
                all_clusters_C = deepcopy(np.append(all_clusters_C,tmp_append,axis = 0))
        elif iter == 3:
            tmp_append = np.empty([max_clus_in_shank - all_clusters_D.shape[0],7],dtype = float)
            tmp_append[:] = np.nan
            if tmp_append.shape[0] != 0:
                all_clusters_D = deepcopy(np.append(all_clusters_D,tmp_append,axis = 0))
    
    # Concatenating (all cluster )
    all_clusters = np.stack((all_clusters_A,all_clusters_B,all_clusters_C,all_clusters_D),axis = 2)
    # total neurons by cell type
    # celltype_total = celltype_excit + celltype_inhib
    # total_activity_act = act_FR
    # work_amount_act = act_FR / act_nclus
    # E/I ratio (network imbalance shown)
    E_I = excitatory_cell/inhibitory_cell
    # Saving mouse summary for averaging 
    full_mouse_ephys = {}
    full_mouse_ephys['List'] = names_datasets
    full_mouse_ephys['act_nclus_total'] = act_nclus_total
    full_mouse_ephys['suppressed_nclus_total'] = suppressed_nclus_total
    full_mouse_ephys['nor_nclus_total'] = nor_nclus
    full_mouse_ephys['N_chan_shank'] = N_chan_shank
    full_mouse_ephys['excitatory_cell'] = excitatory_cell
    full_mouse_ephys['inhibitory_cell'] = inhibitory_cell
    full_mouse_ephys['act_nclus'] = act_nclus
    full_mouse_ephys['act_E_nclus'] = act_E_nclus
    full_mouse_ephys['act_I_nclus'] = act_I_nclus
    full_mouse_ephys['suppressed_nclus'] = suppressed_nclus
    full_mouse_ephys['x_ticks_labels'] = x_ticks_labels
    full_mouse_ephys['celltype_total'] = celltype_total
    full_mouse_ephys['celltype_shank'] = celltype_shank
    full_mouse_ephys['clus_N'] = clus_N
    full_mouse_ephys['activity_nor'] = activity_nor             # Normalized change in number of spikes during stimulation
    full_mouse_ephys['activity_non'] = activity_non             # Change in number of spikes during stimulation
    full_mouse_ephys['activity_non_abs'] = activity_non_abs
    full_mouse_ephys['activity_spont'] = activity_spont         # Average FR over all clusters in each session before stim
    full_mouse_ephys['activity_event'] = activity_event         # Min/Max FR over all clusters in each session during stim
    full_mouse_ephys['all_clusters'] = all_clusters
    full_mouse_ephys['FR_act'] = act_FR
    full_mouse_ephys['spont_FR_avg'] = np.array(avg_spont_FR, dtype='float32')      # Average FR over all clusters in each session before stim
    full_mouse_ephys['stim_FR_avg'] = np.array(avg_stim_FR, dtype='float32')        # Avg Min/Max FR over all clusters in each session during stim
    
    sio.savemat(os.path.join(source_dir,'full_mouse_ephys.mat'), full_mouse_ephys)
    np.savez(os.path.join(source_dir,'full_mouse_T2P.npz'),T2P = np.array(T2P_allsessions,dtype = object))        # saving as object
    main_df.to_pickle(os.path.join(source_dir,'all_cluster.pkl'))           # info for all clusters in this mouse (all sessions). A complete dataframe. No extra info needed
    # sio.savemat(os.path.join(source_dir,'full_mouse_T2P.mat'), T2P_allsessions)
    # np.save(os.path.join(source_dir,'full_mouse_T2P.npy'), T2P_allsessions)     
    # sio.savemat(os.path.join(source_dir,'full_mouse_T2P.mat'), {'T2P_allsessions' : T2P_allsessions})

    # Plot of cell types + activated/suppressed + excitatory/inhibitory neurons
    filename_save = os.path.join(source_dir,'Population_analysis_cell_activation.png')
    f, a = plt.subplots(2,3)
    a[0,0].set_ylabel('Pop. Count')
    a[1,0].set_ylabel('Pop. Count')
    a[1,0].set_xlabel('Days')
    a[1,1].set_xlabel('Days')
    a[1,2].set_xlabel('Days')
    len_str = 'Population Analysis'
    f.suptitle(len_str)
    a[0,0].set_title("Activated Neurons")
    a[0,0].plot(x_ticks_labels,act_nclus[:,0],'r', lw=1.5)
    a[0,0].plot(x_ticks_labels,act_nclus[:,1],'g', lw=1.5)
    a[0,0].plot(x_ticks_labels,act_nclus[:,2],'b', lw=1.5)
    a[0,0].plot(x_ticks_labels,act_nclus[:,3],'y', lw=1.5)
    a[0,0].legend(['ShankA', 'ShankB','ShankC','ShankD'])
    a[1,0].set_title("Suppressed Neurons")
    a[1,0].plot(x_ticks_labels,suppressed_nclus[:,0],'r', lw=1.5)
    a[1,0].plot(x_ticks_labels,suppressed_nclus[:,1],'g', lw=1.5)
    a[1,0].plot(x_ticks_labels,suppressed_nclus[:,2],'b', lw=1.5)
    a[1,0].plot(x_ticks_labels,suppressed_nclus[:,3],'y', lw=1.5)
    a[1,0].legend(['ShankA','ShankB','ShankC', 'ShankD'])
    a[1,1].set_title("All neurons")
    a[1,1].plot(x_ticks_labels,act_nclus_total,'r', lw=1.5)
    a[1,1].plot(x_ticks_labels,suppressed_nclus_total,'b', lw=1.5)
    a[1,1].legend(['Activated','Suppressed'])
    a[0,2].set_title("Excitatory Neurons (Waveform Analysis)")
    a[0,2].plot(x_ticks_labels,excitatory_cell[:,0],'r', lw=1.5)
    a[0,2].plot(x_ticks_labels,excitatory_cell[:,1],'g', lw=1.5)
    a[0,2].plot(x_ticks_labels,excitatory_cell[:,2],'b', lw=1.5)
    a[0,2].plot(x_ticks_labels,excitatory_cell[:,3],'y', lw=1.5)
    a[0,2].legend(['ShankA', 'ShankB','ShankC','ShankD'])
    a[1,2].set_title("Inhibitory Neurons (Waveform Analysis)")
    a[1,2].plot(x_ticks_labels,inhibitory_cell[:,0],'r', lw=1.5)
    a[1,2].plot(x_ticks_labels,inhibitory_cell[:,1],'g', lw=1.5)
    a[1,2].plot(x_ticks_labels,inhibitory_cell[:,2],'b', lw=1.5)
    a[1,2].plot(x_ticks_labels,inhibitory_cell[:,3],'y', lw=1.5)
    a[1,2].legend(['ShankA', 'ShankB','ShankC','ShankD'])
    a[0,1].set_title("Cell Types")
    a[0,1].plot(x_ticks_labels,celltype_total[:,0],'g', lw=1.5)
    a[0,1].plot(x_ticks_labels,celltype_total[:,1],'k', lw=1.5)
    a[0,1].plot(x_ticks_labels,celltype_total[:,2],'y', lw=1.5)
    a[0,1].legend(['Pyramidal','Narrow','Wide'])
    f.set_size_inches((20, 8), forward=False)
    plt.savefig(filename_save,format = 'png')
    plt.close(f)

    # Population Analysis Figure2 (not by shank but global)
    filename_save = os.path.join(source_dir,'Population_analysis_cell_2.png')
    f, a = plt.subplots(1,3)
    a[0].set_ylabel('Pop. Count')
    # a[0,].set_ylabel('Pop. Count')
    a[0].set_xlabel('Days')
    a[1].set_xlabel('Days')
    a[2].set_xlabel('Days')
    len_str = 'Population Analysis'
    f.suptitle(len_str)
    a[0].set_title("All Populations (this mouse)")
    a[0].plot(x_ticks_labels,clus_N[:,0],'r', lw=1.5)
    a[0].plot(x_ticks_labels,clus_N[:,1],'b', lw=1.5)
    a[0].legend(['Suppressed','Activated'])
    a[1].set_title("All Populations (this mouse)")
    a[1].plot(x_ticks_labels,clus_N[:,2],'r', lw=1.5)
    a[1].plot(x_ticks_labels,clus_N[:,3],'b', lw=1.5)
    a[1].legend(['Excitatory','Inhibitory'])
    a[2].set_title("All Populations (this mouse)")
    a[2].plot(x_ticks_labels,clus_N[:,4],'r', lw=1.5)
    a[2].plot(x_ticks_labels,clus_N[:,5],'b', lw=1.5)
    a[2].legend(['Stim-locked E Cells','Stim-locked I Cells'])
    f.set_size_inches((20, 8), forward=False)
    plt.savefig(filename_save,format = 'png')
    plt.close(f)
    # Neural Activity (Spikes)
    filename_save = os.path.join(source_dir,'Neural_Activity_Spikes.png')
    f, a = plt.subplots(2,3)
    a[0,0].set_ylabel(r"$\Delta$ $S_n$")
    a[1,0].set_ylabel(r"$\Delta$ $S$")
    a[1,0].set_xlabel('Days')
    a[1,1].set_xlabel('Days')
    a[1,2].set_xlabel('Days')
    len_str = 'Tracking Neural Activity Post Stroke'
    f.suptitle(len_str)
    a[0,0].set_title("All Activity (this mouse)")
    a[0,0].plot(x_ticks_labels,np.sum(activity_nor[:,0,:],axis = 1),'r', lw=1.5)
    a[0,0].plot(x_ticks_labels,np.sum(activity_nor[:,1,:],axis = 1),'b', lw=1.5)
    a[0,0].legend(['Suppressed','Activated'])
    a[0,1].set_title("All Activity (this mouse)")
    a[0,1].plot(x_ticks_labels,np.sum(activity_nor[:,2,:],axis = 1),'r', lw=1.5)
    a[0,1].plot(x_ticks_labels,np.sum(activity_nor[:,3,:],axis = 1),'b', lw=1.5)
    a[0,1].legend(['Excitatory','Inhibitory'])
    a[0,2].set_title("All Activity (this mouse)")
    a[0,2].plot(x_ticks_labels,np.sum(activity_nor[:,4,:],axis = 1),'r', lw=1.5)
    a[0,2].plot(x_ticks_labels,np.sum(activity_nor[:,5,:],axis = 1),'b', lw=1.5)
    a[0,2].legend(['Activated E Cells','Activated I Cells'])
    a[1,0].set_title("All Activity (this mouse)")
    a[1,0].plot(x_ticks_labels,np.sum(activity_non[:,0,:],axis = 1),'r', lw=1.5)
    a[1,0].plot(x_ticks_labels,np.sum(activity_non[:,1,:],axis = 1),'b', lw=1.5)
    a[1,0].legend(['Suppressed','Activated'])
    a[1,1].set_title("All Activity (this mouse)")
    a[1,1].plot(x_ticks_labels,np.sum(activity_non[:,2,:],axis = 1),'r', lw=1.5)
    a[1,1].plot(x_ticks_labels,np.sum(activity_non[:,3,:],axis = 1),'b', lw=1.5)
    a[1,1].legend(['Excitatory','Inhibitory'])
    a[1,2].set_title("All Activity (this mouse)")
    a[1,2].plot(x_ticks_labels,np.sum(activity_non[:,4,:],axis = 1),'r', lw=1.5)
    a[1,2].plot(x_ticks_labels,np.sum(activity_non[:,5,:],axis = 1),'b', lw=1.5)
    a[1,2].legend(['Stim-locked E Cells','Stim-locked I Cells'])
    f.set_size_inches((20, 8), forward=False)
    plt.savefig(filename_save,format = 'png')
    plt.close(f)
    # Neural Activity (Spikes) by shank
    filename_save = os.path.join(source_dir,'Neural_Activity_Spikes-shank.png')
    f, a = plt.subplots(2,3)
    a[0,0].set_ylabel(r"$\Delta$ $S$")
    a[1,0].set_ylabel(r"$\Delta$ $S$")
    a[1,0].set_xlabel('Days')
    a[1,1].set_xlabel('Days')
    a[1,2].set_xlabel('Days')
    len_str = 'Tracking Neural Activity Post Stroke'
    f.suptitle(len_str)
    a[0,0].set_title("Activity by Shank (suppressed)")
    a[0,0].plot(x_ticks_labels,activity_non[:,0,:], lw=1.5)
    a[0,0].legend(['A','B','C','D'])
    a[0,1].set_title("Activity by Shank (activated)")
    a[0,1].plot(x_ticks_labels,activity_non[:,1,:], lw=1.5)
    a[0,1].legend(['A','B','C','D'])
    a[0,2].set_title("Activity by Shank (E cell)")
    a[0,2].plot(x_ticks_labels,activity_non[:,2,:], lw=1.5)
    a[0,2].legend(['A','B','C','D'])
    a[1,0].set_title("Activity by Shank (I cell)")
    a[1,0].plot(x_ticks_labels,activity_non[:,3,:], lw=1.5)
    a[1,0].legend(['A','B','C','D'])
    a[1,1].set_title("Stimulus Locked E cell")
    a[1,1].plot(x_ticks_labels,activity_non[:,4,:], lw=1.5)
    a[1,1].legend(['A','B','C','D'])
    a[1,2].set_title("Stimulus Locked I cell")
    a[1,2].plot(x_ticks_labels,activity_non[:,5,:], lw=1.5)
    a[1,2].legend(['A','B','C','D'])
    f.set_size_inches((20, 8), forward=False)
    plt.savefig(filename_save,format = 'png')
    plt.close(f)
    # Neural Activity (Bursting)
    
    # Plot for Trough2Peak latency (in ms)
    tmp_str = source_dir.split('/')[-2:]
    tmp_str = ' '.join(tmp_str)
    if tmp_str[-1] == ' ':
        tmp_str = tmp_str[:-1]
    f, ax1 = plt.subplots(1,1)
    filename_save = os.path.join(source_dir,'TP_latency_histogram_' + tmp_str + '.png')
    df_excit,df_inhib = convert2df(T2P_allsessions)
    sns.histplot(data=df_excit, x="T2P", color="red", label="Trough to Peak", kde=True, ax = ax1, kde_kws = {'bw_adjust' : 1.3},binwidth = 0.05)
    sns.histplot(data=df_inhib, x="T2P", color="skyblue", label="Trough to Peak", kde=True, ax = ax1,kde_kws = {'bw_adjust' : 1.7},binwidth = 0.05 )
    ax1.set_xlabel('Trough to Peak (ms)')
    # f = plt.gcf()
    f.set_size_inches((12, 6), forward=False)
    plt.savefig(filename_save,format = 'png')
    plt.close(f)
    # Plots for average FR
    filename_save = os.path.join(source_dir,'FR_analysis_cell_activation.png')
    f, a = plt.subplots(1,2)
    a[0].set_ylabel('Avg FR')
    a[1].set_ylabel('E/I ratio')
    a[0].set_xlabel('Days')
    a[1].set_xlabel('Days')
    len_str = 'FR analysis'
    f.suptitle(len_str)
    a[0].set_title("Avg FR by shank of activated neurons")
    a[0].plot(x_ticks_labels,act_FR[:,0],'r', lw=1.5)
    a[0].plot(x_ticks_labels,act_FR[:,1],'g', lw=1.5)
    a[0].plot(x_ticks_labels,act_FR[:,2],'b', lw=1.5)
    a[0].plot(x_ticks_labels,act_FR[:,3],'y', lw=1.5)
    a[0].legend(['ShankA','ShankB','ShankC', 'ShankD'])
    a[1].set_title("E/I ratio imbalance post stroke")
    a[1].plot(x_ticks_labels,E_I[:,0],'r', lw=1.5)
    a[1].plot(x_ticks_labels,E_I[:,1],'g', lw=1.5)
    a[1].plot(x_ticks_labels,E_I[:,2],'b', lw=1.5)
    a[1].plot(x_ticks_labels,E_I[:,3],'y', lw=1.5)
    a[1].legend(['ShankA','ShankB','ShankC', 'ShankD'])
    f.set_size_inches((12, 6), forward=False)
    plt.savefig(filename_save,format = 'png')
    plt.close(f)


# plt.figure()
# plt.bar(x_ticks_labels, act_nclus[:,0])
# plt.figure()
# plt.bar(x_ticks_labels, act_nclus[:,1])
# plt.figure()
# plt.bar(x_ticks_labels, act_nclus[:,2])
# plt.figure()
# plt.bar(x_ticks_labels, suppressed_nclus[:,0])
# plt.figure()
# plt.bar(x_ticks_labels, suppressed_nclus[:,1])
# plt.figure()
# plt.bar(x_ticks_labels, suppressed_nclus[:,2])




# Extract templates and analyze their cell types

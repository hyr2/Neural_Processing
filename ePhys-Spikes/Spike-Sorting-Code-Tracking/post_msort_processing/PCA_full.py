#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:58:28 2023

@author: hyr2-office
"""

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
    


def PCA_clustering(pca_scores_state,dict_params):    
    dict_pred = {}
    
    # To add: silhouette coeefficient and intertia value
    
    # K-means clustering in the PCA space 
    kmeans = skC.KMeans(n_clusters = dict_params['N_k'])
    y_pred = kmeans.fit_predict(pca_scores_state)
    dict_pred['KMeans'] = y_pred
    
    # DBSCAN clustering in the PCA space
    dbscan = skC.DBSCAN(eps = dict_params['eps_db'], min_samples = dict_params['min_samples'])
    y_pred = dbscan.fit_predict(pca_scores_state)
    dict_pred['DbScan'] = y_pred
    
    # OPTICS clustering in the PCA space
    optics_model = skC.OPTICS(eps=dict_params['eps_opt'], min_samples=dict_params['min_samples'])
    y_pred = optics_model.fit_predict(pca_scores_state)
    dict_pred['Optics'] = y_pred
        
    return dict_pred

def KMeans_evaluate_inertia(k_max,data_pca,dir_save):
    arr_inertia_ = []
    range_n_clusters = np.arange(1,k_max+1)
    for n_clusters in range_n_clusters:
        clusterer = skC.KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        clusterer.fit_predict(data_pca)
        arr_inertia_.append(clusterer.inertia_)
        print(
            "For n_clusters =",
            n_clusters,
            "The average inertia value is :",
            clusterer.inertia_
        )
    arr_inertia = np.array(arr_inertia_)
    fig, ax1 = plt.subplots(1,1)
    fig.set_size_inches(7,7)
    ax1.plot(range_n_clusters, arr_inertia)    
    filename_save= os.path.join(dir_save,'KMeans_inertia.png')
    plt.savefig(filename_save,dpi = 100)

def KMeans_evaluate_silhouette(k_max,data_pca,dir_save):
    output_arr = []
    range_n_clusters = np.arange(2,k_max+1)
    arr_silhoutte_ = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data_pca) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = skC.KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(data_pca)
        output_arr.append(cluster_labels)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = skM.silhouette_score(data_pca, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg
        )
        arr_silhoutte_.append(silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = skM.silhouette_samples(data_pca, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            data_pca[:, 0], data_pca[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        #plt.show()
        filename_save= os.path.join(dir_save,f'{n_clusters}_KMeans_Silhouette.png')
        plt.savefig(filename_save,dpi = 100)
        
    fg,ax = plt.subplots(1,1)
    fg.set_size_inches(7,7)
    ax.plot(range_n_clusters,np.array(arr_silhoutte_))
    filename_save= os.path.join(dir_save,'KMeans_Silhouette_Avg.png')
    plt.savefig(filename_save,dpi = 100)
        
    
    return range_n_clusters,output_arr

def fig5_e(input_list, filename_save,  df_main, dict_config):
    
    def fill_nans_single(input_list,local_time,local_count,prev_count):
        # ideal time axis
        time_ideal = np.array([-3,-2, 2, 7 , 14, 21, 28, 35, 42, 49, 56],dtype = float)
        # filling nans rh3
        
        indx_arr_nan = np.logical_not(np.in1d(time_ideal,local_time))
        local_time_new = copy.deepcopy(time_ideal)
        local_time_new[indx_arr_nan] = np.nan
        temp_arr = np.empty(time_ideal.shape,dtype = float)
        temp_arr[:] = np.nan
        indx_insert = np.where(np.logical_not(np.isnan(local_time_new)))[0]
        cluster_range = np.arange(prev_count,prev_count + local_count,1)
        for iter_i in cluster_range:
            arr_tmp = copy.deepcopy(temp_arr)
            arr_tmp[indx_insert] = input_list[iter_i]
            input_list[iter_i] = arr_tmp
        return input_list
    
    cum_sum = np.array([0, dict_config['rh3_count'],dict_config['bc7_count'],dict_config['rh8_count'],dict_config['rh11_count']])
    cum_sum = np.cumsum(cum_sum)
    input_list = fill_nans_single(input_list , dict_config['rh3_time'], dict_config['rh3_count'],cum_sum[0])     # rh3
    input_list = fill_nans_single(input_list , dict_config['bc7_time'], dict_config['bc7_count'],cum_sum[1])     # bc7
    input_list = fill_nans_single(input_list , dict_config['rh8_time'], dict_config['rh8_count'],cum_sum[2])     # rh8
    input_list = fill_nans_single(input_list , dict_config['rh11_time'], dict_config['rh11_count'],cum_sum[3])     # rh11
    
    # input_list 
    time_ideal = np.array([-3,-2, 2, 7 , 14, 21, 28, 35, 42, 49, 56],dtype = float)
    full_array = np.zeros([len(input_list),time_ideal.size],dtype = float)
    for iter_i in range(len(input_list)):
        full_array[iter_i,:] = input_list[iter_i]
    

    
    # All
    x_bar = np.array([0,1])
    N_norm = full_array.shape[0]
    y_bar = np.array([np.mean(full_array[:,0:2]),np.nanmean(full_array[:,5:])])
    y_std = np.array([np.nanstd(full_array[:,0:2]),np.nanstd(full_array[:,5:])])
    y_sem = y_std / np.sqrt(2*N_norm)
    fig,ax = plt.subplots(1,2)
    ax = ax.flatten()
    ax[0].plot(time_ideal,np.nanmean(full_array,axis = 0))
    ax[1].bar(x_bar,y_bar,width = 0.9,align = 'center',color = 'k',alpha = 0.6)
    ax[1].errorbar(x_bar,y_bar,y_sem,fmt='None',linewidth = 2.4,ecolor = 'k',elinewidth=2.5)
    plt.savefig(filename_save+'_all.png')        
    
    # Celltype
    mask_P = (df_main['type'] == 'P').to_numpy()
    full_array_P = full_array[mask_P,:]
    mask_NI = (df_main['type'] == 'NI').to_numpy()
    full_array_NI = full_array[mask_NI,:]
    
    N_norm = full_array_P.shape[0]
    y_bar = np.array([np.mean(full_array_P[:,0:2]),np.nanmean(full_array_P[:,5:])])
    y_std = np.array([np.nanstd(full_array_P[:,0:2]),np.nanstd(full_array_P[:,5:])])
    y_sem = y_std / np.sqrt(2*N_norm)
    fig,ax = plt.subplots(1,2)
    ax = ax.flatten()
    ax[0].bar(x_bar,y_bar,width = 0.9,align = 'center',color = 'k',alpha = 0.6)
    ax[0].errorbar(x_bar,y_bar,y_sem,fmt='None',linewidth = 2.4,ecolor = 'k',elinewidth=2.5)
    N_norm = full_array_NI.shape[0]
    y_bar = np.array([np.mean(full_array_NI[:,0:2]),np.nanmean(full_array_NI[:,5:])])
    y_std = np.array([np.nanstd(full_array_NI[:,0:2]),np.nanstd(full_array_NI[:,5:])])
    y_sem = y_std / np.sqrt(2*N_norm)
    ax[1].bar(x_bar,y_bar,width = 0.9,align = 'center',color = 'k',alpha = 0.6)
    ax[1].errorbar(x_bar,y_bar,y_sem,fmt='None',linewidth = 2.4,ecolor = 'k',elinewidth=2.5)
    plt.savefig(filename_save+'_celltype.png')    
    
    # By PCA Class
    mask_P = (df_main['PCA_label'] == 0).to_numpy()
    full_array_P = full_array[mask_P,:]
    mask_N = (df_main['PCA_label'] == 1).to_numpy()
    full_array_N = full_array[mask_N,:]
    mask_NC = (df_main['PCA_label'] == 2).to_numpy()
    full_array_NC = full_array[mask_NC,:]

    N_norm = full_array_P.shape[0]
    y_bar = np.array([np.mean(full_array_P[:,0:2]),np.nanmean(full_array_P[:,5:])])
    y_std = np.array([np.nanstd(full_array_P[:,0:2]),np.nanstd(full_array_P[:,5:])])
    y_sem = y_std / np.sqrt(2*N_norm)
    fig,ax = plt.subplots(1,3)
    ax = ax.flatten()
    ax[0].bar(x_bar,y_bar,width = 0.9,align = 'center',color = 'k',alpha = 0.6)
    ax[0].errorbar(x_bar,y_bar,y_sem,fmt='None',linewidth = 2.4,ecolor = 'k',elinewidth=2.5)
    N_norm = full_array_N.shape[0]
    y_bar = np.array([np.mean(full_array_N[:,0:2]),np.nanmean(full_array_N[:,5:])])
    y_std = np.array([np.nanstd(full_array_N[:,0:2]),np.nanstd(full_array_N[:,5:])])
    y_sem = y_std / np.sqrt(2*N_norm)
    ax[1].bar(x_bar,y_bar,width = 0.9,align = 'center',color = 'k',alpha = 0.6)
    ax[1].errorbar(x_bar,y_bar,y_sem,fmt='None',linewidth = 2.4,ecolor = 'k',elinewidth=2.5)
    N_norm = full_array_NC.shape[0]
    y_bar = np.array([np.mean(full_array_NC[:,0:2]),np.nanmean(full_array_NC[:,5:])])
    y_std = np.array([np.nanstd(full_array_NC[:,0:2]),np.nanstd(full_array_NC[:,5:])])
    y_sem = y_std / np.sqrt(2*N_norm)
    ax[2].bar(x_bar,y_bar,width = 0.9,align = 'center',color = 'k',alpha = 0.6)
    ax[2].errorbar(x_bar,y_bar,y_sem,fmt='None',linewidth = 2.4,ecolor = 'k',elinewidth=2.5)
    plt.savefig(filename_save+'_PCA.png')    
    
    return df_main
    
    

def fig5_b(input_list, filename_save):
    sc_bsl = []
    sc_chr = []
    for iter_i in range(len(input_list)):
        sc_bsl.append(np.mean(input_list[iter_i][0:2]))     # days prestroke
        sc_chr.append(np.mean(input_list[iter_i][3:]))  # days > 21
    sc_bsl = np.array(sc_bsl,dtype = float)
    sc_chr = np.array(sc_chr,dtype = float)
    fig,ax = plt.subplots(1,2)
    ax = ax.flatten()
    ax[0].scatter(sc_bsl,sc_chr,s = 20)
    ax[0].set_xlabel('Pre-Stroke')
    ax[0].set_ylabel('Post-Stroke')
    ax[1].scatter(sc_bsl,sc_chr,s = 20)
    ax[1].set_xlim([-10,60])
    ax[1].set_ylim([-10,240])
    ax[1].set_xlabel('Pre-Stroke')
    ax[1].set_ylabel('Post-Stroke')
    plt.savefig(filename_save)
    

def PCA_apply(dict_params,Num_com):
    
    # Hard coded for now
    dict_config =  {}
    dict_config['rh3_time'] = np.array([-3,-2,14,21,28,49])
    dict_config['bc7_time'] = np.array([-3,-2,14,21,28,42])
    dict_config['rh8_time'] = np.array([-3,-2,14,21,28,35,42,49,56])
    dict_config['rh11_time'] = np.array([-3,-2,14,21,28,35,42,49])
    # dict_config['rh7_time'] = np.array([-3,-2,14,21,28,35,42,49])
    
    # Save folder:
    output_folder = os.path.join('/home/hyr2/Documents/Data/NVC/Results/PCA_analysis')
    
    # For Plasticity Metrics (Z-scored thresholded Z > 2 and Z < -0.5)
    
    dict_config['rh3_count'] = c_mouse_rh3.size
    dict_config['bc7_count'] = c_mouse_bc7.size
    dict_config['rh8_count'] = c_mouse_rh8.size
    dict_config['rh11_count'] = c_mouse_rh11.size
    # dict_config['rh7_count'] = 
    
    
    c_all_mouse = np.concatenate((c_mouse_rh3,c_mouse_bc7,c_mouse_rh8,c_mouse_rh11))    # concatenate all
    c_mouse_rh3 = np.load(os.path.join('/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh3/','all_shanks_clus_property_processed.npy'),allow_pickle=True)
    c_mouse_bc7 = np.load(os.path.join('/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_bc7/','all_shanks_clus_property_processed.npy'),allow_pickle=True)
    c_mouse_rh8 = np.load(os.path.join('/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh8/','all_shanks_clus_property_processed.npy'),allow_pickle=True)
    c_mouse_rh11 = np.load(os.path.join('/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/','all_shanks_clus_property_processed.npy'),allow_pickle=True)
    plasticity_data_list_new = []
    spont_FR_data_list = []
    spont_total_data_list = []
    stim_FR_data_list = []
    stim_total_data_list = []
    for iter_i in range(c_all_mouse.size):
        plasticity_data_list_new.append(c_all_mouse[iter_i]['plasticity_metric'])
        spont_FR_data_list.append(c_all_mouse[iter_i]['FR_avg_spont'])              # FR per trial
        spont_total_data_list.append(c_all_mouse[iter_i]['S_total_spont'])          # total spikes per trial
        stim_FR_data_list.append(c_all_mouse[iter_i]['FR_avg_stim'])          # total spikes per trial
        stim_total_data_list.append(c_all_mouse[iter_i]['S_total_stim'])          # total spikes per trial
        
        
    df_main = pd.DataFrame([],columns = ['cluster_id','spont_FR','spont_S','stim_FR','stim_S','type','Z_label','PCA_label'])
    # fig5_e(spont_total_data_list,df_main,dict_config)
    # fig5_e(spont_FR_data_list,df_main,dict_config)
    
    plasticity_data_list_new = np.array(plasticity_data_list_new, dtype = np.int8)
    
    # For Cell Type
    c_mouse_rh3 = np.load(os.path.join('/home/hyr2/Documents/Data/NVC/Tracking/processed_data_rh3/','all_shanks_celltype_processed.npy'))
    c_mouse_bc7 = np.load(os.path.join('/home/hyr2/Documents/Data/NVC/Tracking/processed_data_bc7/','all_shanks_celltype_processed.npy'))
    c_mouse_rh8 = np.load(os.path.join('/home/hyr2/Documents/Data/NVC/Tracking/processed_data_rh8/','all_shanks_celltype_processed.npy'))
    c_mouse_rh11 = np.load(os.path.join('/home/hyr2/Documents/Data/NVC/Tracking/processed_data_rh11/','all_shanks_celltype_processed.npy'))
    
    celltype_data_list_new = []
    celltype_data_list_new.extend(c_mouse_rh3.tolist())
    celltype_data_list_new.extend(c_mouse_bc7.tolist())
    celltype_data_list_new.extend(c_mouse_rh8.tolist())
    celltype_data_list_new.extend(c_mouse_rh11.tolist())
    
    celltype_data_list_new = np.array(celltype_data_list_new,dtype = str)
    df_main['type'] = celltype_data_list_new
    # celltype_data_list_edgecolors = copy.deepcopy(celltype_data_list_new)
    # celltype_data_list_edgecolors[celltype_data_list_edgecolors == 'NI'] = 0
    # celltype_data_list_edgecolors[celltype_data_list_edgecolors == 'P'] = 1
    # celltype_data_list_edgecolors[celltype_data_list_edgecolors == 'WI'] = 1
    # celltype_data_list_edgecolors = celltype_data_list_edgecolors.astype(np.int8)
    
    
    # For PCA
    mouse_rh3 = np.load(os.path.join('/home/hyr2/Documents/Data/NVC/Tracking/processed_data_rh3/','all_shanks_pca_preprocessed.npy'),allow_pickle=True)
    mouse_bc7 = np.load(os.path.join('/home/hyr2/Documents/Data/NVC/Tracking/processed_data_bc7/','all_shanks_pca_preprocessed.npy'),allow_pickle=True)
    mouse_rh8 = np.load(os.path.join('/home/hyr2/Documents/Data/NVC/Tracking/processed_data_rh8/','all_shanks_pca_preprocessed.npy'),allow_pickle=True)
    mouse_rh11 = np.load(os.path.join('/home/hyr2/Documents/Data/NVC/Tracking/processed_data_rh11/','all_shanks_pca_preprocessed.npy'),allow_pickle=True)


    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder,'all_units_FR'))

    scaled_data_list_new = np.concatenate((mouse_rh3,mouse_bc7,mouse_rh8,mouse_rh11),axis = 1)
    scaled_data_list_new = np.delete(scaled_data_list_new,np.array([1,2]),axis=0)       # keeping only standard scaler
    
    time_axis = np.arange(scaled_data_list_new.shape[2])
    # # Plotting all units
    # for iter_i in range(scaled_data_list_new.shape[1]):
    #     y_data =  scaled_data_list_new[0,iter_i,:]
    #     fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
    #     axes.plot(time_axis, y_data, color='gold')
    #     axes.set_xlabel('Stroke Phases')
    #     axes.set_ylabel('FR (arb.)')
    #     plt.savefig(os.path.join(output_folder,'all_units_FR',f'{iter_i}_Tracked_Unit_.png'))
    #     plt.close(fig)

    
    for iter_l in range(len(scaled_data_list_new)):
        
        raw_data = scaled_data_list_new[iter_l,:,:]
        
        n_clusters = raw_data.shape[1]
        
        fd_local = skfda.FDataGrid(     # This is N x M X q (observations x features x 1)
            data_matrix = raw_data,
            grid_points = None
            )
        
        # Performing PCA for non functional
        fig, axes = plt.subplots(2,1, figsize=(10,12), dpi=100)
        axes = axes.flatten()
        
        fpca_discretized = FPCA(n_components=Num_com)
        fpca_discretized.fit(fd_local)
        components_ = fpca_discretized.components_
        components_ = components_.data_matrix
        axes[0].plot(np.squeeze(components_.T),linewidth = 2.9)
        axes[0].legend(['pc1','pc2','pc3'])
        
        pca_scores_state = fpca_discretized.transform(fd_local)
        lst_col = ['pc'+str(iter_ll+1) for iter_ll in range(pca_scores_state.shape[1])]
        df_pca_scores = pd.DataFrame(pca_scores_state,columns=lst_col)
        PC = range(1, Num_com+1)
        axes[1].bar(PC, 100*fpca_discretized.explained_variance_ratio_, color='gold')
        axes[1].set_xlabel('Principal Components')
        axes[1].set_ylabel('Variance %')
        axes[1].set_xticks(PC)
        plt.savefig(os.path.join(output_folder,f'{iter_l}_nonfunctional.png'))
        
        if Num_com == 3:
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            ax.scatter3D(pca_scores_state[:,0],pca_scores_state[:,1],pca_scores_state[:,2])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.savefig(os.path.join(output_folder,f'{iter_l}_nonfunc_scatter.png'))
        elif Num_com == 2:
            fig,ax = plt.subplots(1,1)
            ax.scatter(pca_scores_state[:,0],pca_scores_state[:,1],s = 4)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.savefig(os.path.join(output_folder,f'{iter_l}_nonfunc_scatter.png'))
        
        
        
        # Performing PCA functional
        # fig, axes = plt.subplots(2,1, figsize=(10,12), dpi=100)
        # axes = axes.flatten()
        
        basis = skfda.representation.basis.BSplineBasis(n_basis=9)
        basis_fd = fd_local.to_basis(basis)
        basis_fd_plot = basis_fd.to_basis()
            
        fpca = FPCA(n_components=Num_com)
        fpca.fit(basis_fd)
        components_ = fpca.components_.plot(linewidth = 2.9)
        plt.legend(['pc1','pc2','pc3'])
        plt.savefig(os.path.join(output_folder,f'{iter_l}_functional_comp.png'))
        
        pca_scores_state = fpca.transform(basis_fd)
        lst_col = ['pc'+str(iter_ll+1) for iter_ll in range(pca_scores_state.shape[1])]
        df_pca_scores = pd.DataFrame(pca_scores_state,columns=lst_col)
        PC = range(1, Num_com +1)
        fig, axes = plt.subplots(1,1, figsize=(10,12), dpi=100)
        axes.bar(PC, 100*fpca.explained_variance_ratio_, color='gold')
        axes.set_xlabel('Principal Components')
        axes.set_ylabel('Variance %')
        axes.set_xticks(PC)
        plt.savefig(os.path.join(output_folder,f'{iter_l}_functional_variance.png'))
        
        dict_unit_labels = PCA_clustering(pca_scores_state, dict_params) 
        
        # KMeans labelled plot
        # colors = cm.nipy_spectral(dict_unit_labels['KMeans'].astype(float) / dict_params['N_k'])
        cmap = plt.get_cmap("Spectral")
        colors = cmap(dict_unit_labels['KMeans'].astype(float) / dict_params['N_k'])
        ct_mask = np.vstack((celltype_data_list_new == 'NI',np.logical_or(celltype_data_list_new == 'P',celltype_data_list_new =='WI'))) # celltype masks
        if Num_com == 3:
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            ax.scatter3D(pca_scores_state[ct_mask[0,:],0],pca_scores_state[ct_mask[0,:],1],pca_scores_state[ct_mask[0,:],2],s = 60, c = colors[ct_mask[0,:]],marker = 'o')
            ax.scatter3D(pca_scores_state[ct_mask[1,:],0],pca_scores_state[ct_mask[1,:],1],pca_scores_state[ct_mask[1,:],2],s = 60, c = colors[ct_mask[1,:]],marker = '^')
            # ax.scatter3D(pca_scores_state[ct_mask[2,:],0],pca_scores_state[ct_mask[2,:],1],pca_scores_state[ct_mask[2,:],2],s = 60, c = colors[ct_mask[2,:]],marker = 'D')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.savefig(os.path.join(output_folder,f'{iter_l}_func_scatter_KMeans.png'))
        elif Num_com == 2:
            fig,ax = plt.subplots(1,1)
            ax.scatter(pca_scores_state[ct_mask[0,:],0],pca_scores_state[ct_mask[0,:],1],s = 60,c = colors[ct_mask[0,:]],marker = 'o')
            ax.scatter(pca_scores_state[ct_mask[1,:],0],pca_scores_state[ct_mask[1,:],1],s = 60,c = colors[ct_mask[1,:]],marker = '^')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.savefig(os.path.join(output_folder,f'{iter_l}_func_scatter_KMeans.png'))
        # DbScan labelled plot
        cmap = plt.get_cmap("Spectral")
        colors = cmap(dict_unit_labels['DbScan'].astype(float) / dict_params['N_k'])
        ct_mask = np.vstack((celltype_data_list_new == 'NI',np.logical_or(celltype_data_list_new == 'P',celltype_data_list_new =='WI'))) # celltype masks
        if Num_com == 3:
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            ax.scatter3D(pca_scores_state[ct_mask[0,:],0],pca_scores_state[ct_mask[0,:],1],pca_scores_state[ct_mask[0,:],2],s = 60, c = colors[ct_mask[0,:]],marker = 'o')
            ax.scatter3D(pca_scores_state[ct_mask[1,:],0],pca_scores_state[ct_mask[1,:],1],pca_scores_state[ct_mask[1,:],2],s = 60, c = colors[ct_mask[1,:]],marker = '^')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.savefig(os.path.join(output_folder,f'{iter_l}_func_scatter_DbScan.png'))
        elif Num_com == 2:
            fig,ax = plt.subplots(1,1)
            ax.scatter(pca_scores_state[ct_mask[0,:],0],pca_scores_state[ct_mask[0,:],1],s = 60,c = colors[ct_mask[0,:]],marker = 'o')
            ax.scatter(pca_scores_state[ct_mask[1,:],0],pca_scores_state[ct_mask[1,:],1],s = 60,c = colors[ct_mask[1,:]],marker = '^')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.savefig(os.path.join(output_folder,f'{iter_l}_func_scatter_DbScan.png'))
        # OPTICS labelled plot
        cmap = plt.get_cmap("Spectral")
        colors = cmap(dict_unit_labels['Optics'].astype(float) / dict_params['N_k'])
        ct_mask = np.vstack((celltype_data_list_new == 'NI',np.logical_or(celltype_data_list_new == 'P',celltype_data_list_new =='WI'))) # celltype masks
        if Num_com == 3:
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            ax.scatter3D(pca_scores_state[ct_mask[0,:],0],pca_scores_state[ct_mask[0,:],1],pca_scores_state[ct_mask[0,:],2],s = 60, c = colors[ct_mask[0,:]],marker = 'o')
            ax.scatter3D(pca_scores_state[ct_mask[1,:],0],pca_scores_state[ct_mask[1,:],1],pca_scores_state[ct_mask[1,:],2],s = 60, c = colors[ct_mask[1,:]],marker = '^')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.savefig(os.path.join(output_folder,f'{iter_l}_func_scatter_optics.png'))
        elif Num_com == 2:
            fig,ax = plt.subplots(1,1)
            ax.scatter(pca_scores_state[ct_mask[0,:],0],pca_scores_state[ct_mask[0,:],1],s = 60,c = colors[ct_mask[0,:]],marker = 'o')
            ax.scatter(pca_scores_state[ct_mask[1,:],0],pca_scores_state[ct_mask[1,:],1],s = 60,c = colors[ct_mask[1,:]],marker = '^')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.savefig(os.path.join(output_folder,f'{iter_l}_func_scatter_optics.png'))
            
        plt.close('all')
        
        # Evaluating KMeans algorithm for various K
        dir_save = os.path.join(output_folder,'KMeans_eval')
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        # KMeans_evaluate_inertia(8,pca_scores_state,dir_save)
        # range_n_clusters, list_cluster_labels = KMeans_evaluate_silhouette(5,pca_scores_state,dir_save)
        time_axis = fd_local.grid_points[0]
        # for iter_ll in range(range_n_clusters.shape[0]):
        #     n_clusters = range_n_clusters[iter_ll]
        #     cluster_labels = list_cluster_labels[iter_ll]
            
        #     # Plotting the basis functions fitted on the raw data
        #     for iter_i in range(scaled_data_list_new.shape[1]):
        #         fig, ax = plt.subplots(1,1, figsize=(10,12), dpi=100)
        #         basis_fd_plot[iter_i].plot(axes=ax)
        #         ax.set_xlabel('Stroke Phases')
        #         ax.set_ylabel('FR (arb.)')
        #         fig.suptitle(f'{cluster_labels[iter_i]}')
        #         plt.savefig(os.path.join(output_folder,'all_units_FR',f'K_{iter_ll}_{iter_i}_Tracked_Unit_.png'))
        #         plt.close(fig)
            
        #     fig,ax = plt.subplots(1,n_clusters,figsize = (14,4))
        #     ax = ax.flatten()
        #     for iter_lll in range(n_clusters):    
        #         raw_data_local = raw_data[cluster_labels == iter_lll , :]
        #         raw_data_local = np.mean(raw_data_local, axis = 0)
        #         ax[iter_lll].plot(time_axis,raw_data_local)
        #         ax[iter_lll].set_ylabel('Z-Score')
        #         ax[iter_lll].set_xlabel('Normalized Stroke Timeline')
        #     plt.savefig(os.path.join(dir_save,f'{iter_ll}_KMeans_AvgRawWaveform_allClusters.png'),dpi = 100,format = 'png')                
        cluster_labels = dict_unit_labels['KMeans'] 
        raw_data_local = raw_data[cluster_labels == 1 , :]  # testing label 1
        raw_data_local = np.mean(raw_data_local, axis = 0)
        plt.plot(time_axis,raw_data_local)
        # Plotting Pie chart and statistics of clusters
        data_Z = [(plasticity_data_list_new == -1).sum(), (plasticity_data_list_new == 1).sum(), (plasticity_data_list_new == 0).sum()]
        data_uml = [(dict_unit_labels['KMeans'] == 1).sum(),(dict_unit_labels['KMeans'] == 0).sum(),(dict_unit_labels['KMeans'] == 2).sum()]
        print('Z-scored: ', data_Z)
        print('UML: ', data_uml)
        df_main['Z_label'] = plasticity_data_list_new
        df_main['PCA_label'] = dict_unit_labels['KMeans']
        

        plt.figure()
        labels1 = ['Down-regulated','Up-regulated','Recovered']
        plt.pie(data_Z,labels=labels1)
        plt.savefig(os.path.join(dir_save,'Plasticity_counts_pie.png'),format = 'png',dpi = 300)
        plt.figure()
        labels1 = ['Down-regulated','Up-regulated','Recovered']
        plt.pie(data_uml,labels=labels1)
        plt.savefig(os.path.join(dir_save,'UML_counts_pie.png'),format = 'png',dpi = 300)
        
        # (dict_unit_labels['KMeans'] == 0).sum()     # Down-regulated
        # (dict_unit_labels['KMeans'] == 1).sum()     # Up-regula.red
        # (dict_unit_labels['KMeans'] == 2).sum()     # No change
        
        # For Scatter plots (Fig 5b)
        
        filename_save = os.path.join(output_folder,'avgFR_fig5_b.png')
        fig5_b(spont_FR_data_list,filename_save)
        filename_save = os.path.join(output_folder,'totalS_fig5_b.png')
        fig5_b(spont_total_data_list,filename_save)
        filename_save = os.path.join(output_folder,'avgFR_fig5_e')
        fig5_e(spont_FR_data_list,filename_save,df_main,dict_config)
        filename_save = os.path.join(output_folder,'totalS_fig5_e')
        fig5_e(spont_total_data_list,filename_save,df_main,dict_config)
        filename_save = os.path.join(output_folder,'avgFRStim_fig5_b.png')
        fig5_b(stim_FR_data_list,filename_save)
        filename_save = os.path.join(output_folder,'totalSStim_fig5_b.png')
        fig5_b(stim_total_data_list,filename_save)
        filename_save = os.path.join(output_folder,'avgFRStim_fig5_e')
        fig5_e(stim_FR_data_list,filename_save,df_main,dict_config)
        filename_save = os.path.join(output_folder,'totalSStim_fig5_e')
        fig5_e(stim_total_data_list,filename_save,df_main,dict_config)
        
if __name__ == '__main__':
    
    # Performing clustering 
    dict_params = {}
    dict_params['N_k'] = 3
    dict_params['eps_db'] = 0.58
    dict_params['eps_opt'] = 0.43
    dict_params['min_samples'] = 30
    
    PCA_apply(dict_params,Num_com=3)
    # plt.close('all')
    


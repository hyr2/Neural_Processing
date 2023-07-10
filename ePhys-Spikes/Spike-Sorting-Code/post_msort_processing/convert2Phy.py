#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:57:09 2023

@author: hyr2-office
"""

# This standalone script is used to convert the curated output files from 
# discard_noise_viz.py file and give the correct output files to be used for
# manual curation in the software PHY

import os, sys, json, shutil
from time import time
sys.path.append(os.path.join(os.getcwd(),'utils'))
sys.path.append(os.getcwd())
sys.path.append('../preprocess_rhd') 
from itertools import groupby
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy.signal as signal
import pandas as pd
from utils.Support import read_stimtxt
from utils.read_mda import readmda
from utils.mdaio import writemda64
from Support import calc_key_metrics, calc_merge_candidates, makeSymmetric

# session_folder = '/home/hyr2-office/Documents/Data/NVC/RH-7_REJECTED/10-27-22/'

def func_convert2Phy(session_folder):
    output_phy = os.path.join(session_folder,'phy_output')
    if not os.path.exists(output_phy):
        os.makedirs(output_phy)
    
    
    # Everyting except PCA (feature space)
    firings = readmda(os.path.join(session_folder, "firings_clean.mda")).astype(np.int64)
    template_waveforms = readmda(os.path.join(session_folder, "templates_clean.mda")).astype(np.float64)
    
    
    nSpikes = firings.shape[1]  # number of spikes in the sessions
    
    prim_ch = firings[0,:]   
    spike_times = firings[1,:].astype(dtype = 'uint64')         # This is spike_times.npy
    spike_clusters = firings[2,:].astype(dtype = 'uint32') - 1  # This is spike_clusters.npy
    
    nTemplates = np.max(spike_clusters) + 1                      # number of clusters found
    nChan = np.amax(prim_ch)                                  # number of channels in the recording (effective)
    
    # Needs to be fixed (read all_waveforms_by_cluster.npz)
    amplitudes = np.zeros([nSpikes,])
    all_waveforms_by_cluster = np.load(os.path.join(session_folder,'all_waveforms_by_cluster_clean.npz'))
    for i_clus in range(nTemplates):
        waveforms_this_cluster = all_waveforms_by_cluster['clus%d'%(i_clus+1)]  # cluster IDs in .npz start from 1; (n_spikes, len_waveforms)
        waveform_peaks = np.max(waveforms_this_cluster, axis=1)
        waveform_troughs = np.min(waveforms_this_cluster, axis=1)
        tmp_amp_series = (waveform_peaks-waveform_troughs) * (1-2*(waveform_peaks<0))
        indx_amplitudes = np.where(spike_clusters == i_clus)[0]     # spike_clusters starts from 0
        # assert tmp_amp_series.shape[0] == indx_amplitudes.shape[0], \
        if (tmp_amp_series.shape[0] != indx_amplitudes.shape[0]):       # @jiaaoz please take a look at this error
            print('Number of spikes in cluster %d did not match'%(i_clus+1))
            tmp_num = indx_amplitudes.shape[0] - tmp_amp_series.shape[0] 
            for iter in range(tmp_num):
                tmp_amp_series = np.append(tmp_amp_series,np.mean(tmp_amp_series))
            # print(tmp_amp_series.shape[0] - indx_amplitudes.shape[0])
        amplitudes[indx_amplitudes] = tmp_amp_series
        
    amplitudes = amplitudes.astype(dtype = 'float64')
    # amplitudes = np.ones([nSpikes,],dtype = 'float64')         # This is amplitudes.npy
    
    template_waveforms = np.moveaxis(template_waveforms,[0,1,2],[2,1,0]).astype(dtype = 'float32') # This is templates.npy
    
    spike_templates = spike_clusters                         # This is spike_templates.npy
    
    templates_ind = np.ones([nTemplates,nChan],dtype = 'float64')       
    tmp_arr = np.arange(0,nChan,1)
    templates_ind = templates_ind * np.transpose(tmp_arr)    # This is templates_ind.npy
    
    channel_map = tmp_arr                                    # This is channel_map.npy 
    channel_positions = pd.read_csv(os.path.join(session_folder,'geom.csv'), header=None).values
    channel_positions = channel_positions.astype(dtype = 'float64') # This is channel_positions.npy
    
    # Creating the params.py
    
    file_pre_ms = os.path.join(session_folder,'pre_MS.json')
    with open(file_pre_ms, 'r') as f:
      data_pre_ms = json.load(f)
    F_SAMPLE = float(data_pre_ms['SampleRate'])
    
    fname = os.path.join(output_phy,'params.py')
    path_data = "'empty str'"
    l1 = 'dat_path = ' + path_data + '\n'
    l2 = 'n_channels_dat = ' + str(nChan) + '\n'
    l3 = 'dtype = ' + "'int16'" + '\n'
    l4 = 'offset = ' + str(0) + '\n'
    l5 = 'sample_rate = ' + str(F_SAMPLE) + '\n'
    l6 = 'hp_filtered = ' + str('False') + '\n'
    
    with open(fname, 'w') as f:
        f.writelines([l1,l2,l3,l4,l5,l6])
    
    # Applying the curation mask (accept_mask.csv)
    # curation_mask_file = os.path.join(session_folder,'accept_mask.csv')
    # curation_mask = pd.read_csv(curation_mask_file, header=None, index_col=False,dtype = bool)
    # curation_mask_np = curation_mask.to_numpy()
    # curation_mask_np = np.reshape(curation_mask_np,[curation_mask_np.shape[0],])
    # tmp = curation_mask_np[spike_clusters]    # remove these spikes from the firings.mda structure
    spike_clusters_new = spike_clusters
    # spike_templates_new = deepcopy(spike_templates[tmp])
    spike_times_new = spike_times
    amplitudes_new = amplitudes
    templates_ind_new = templates_ind
    template_waveforms_new = template_waveforms
    nTemplates_new = np.unique(spike_clusters).shape[0]
    new_cluster_id = np.arange(0,nTemplates_new,1)
    old_cluster_id = np.unique(spike_clusters_new)
    # cluster_mapping = old_cluster_id
    for iter in range(nTemplates_new):
        indx_temp = np.where(spike_clusters_new == old_cluster_id[iter])[0]
        spike_clusters_new[indx_temp] = iter
    spike_templates_new = spike_clusters_new
    cluster_mapping = np.vstack((new_cluster_id,old_cluster_id))
    cluster_mapping = np.transpose(cluster_mapping)
    pd.DataFrame(data=cluster_mapping.astype(int)).to_csv(os.path.join(output_phy, "cluster_mapping.csv"), index=False, header=False)
    # spike_templates_new = spike_clusters_new
    
    # Generating similar_templates.npy containing possible merging candidates (Jiaao's code)
    prim_ch_new = prim_ch
    firings_new = np.vstack((prim_ch_new,spike_times_new,spike_clusters_new))
    templates_sliced, _, peak_amplitudes, clus_coordinates = calc_key_metrics(np.moveaxis(template_waveforms_new,[0,1,2],[2,1,0]), firings_new, channel_positions, F_SAMPLE)
    clus_coordinates = pd.read_csv(os.path.join(session_folder,'clus_locations_clean.csv'), header = None).to_numpy()
    merge_cand_mat = calc_merge_candidates(templates_sliced, clus_coordinates, peak_amplitudes)
    merge_cand_mat[:,0] = merge_cand_mat[:,0] - 1   # Cluster IDs now start from 0
    merge_cand_mat[:,1] = merge_cand_mat[:,1] - 1   # Cluster IDs now start from 0
    similar_templates = np.zeros([nTemplates_new,nTemplates_new],dtype = 'single')
    for iter in range(merge_cand_mat.shape[0]):
        similar_templates[merge_cand_mat[iter,0],merge_cand_mat[iter,1]] = merge_cand_mat[iter,2]
    similar_templates_new = deepcopy(makeSymmetric(similar_templates))
    
    np.save(os.path.join(output_phy,'spike_times.npy'),spike_times_new)
    np.save(os.path.join(output_phy,'spike_clusters.npy'),spike_clusters_new)
    np.save(os.path.join(output_phy,'amplitudes.npy'),amplitudes_new)
    np.save(os.path.join(output_phy,'templates.npy'),template_waveforms_new)
    np.save(os.path.join(output_phy,'spike_templates.npy'),spike_templates_new)
    np.save(os.path.join(output_phy,'templates_ind.npy'),templates_ind_new)
    np.save(os.path.join(output_phy,'channel_map.npy'),channel_map)
    np.save(os.path.join(output_phy,'channel_positions.npy'),channel_positions)
    np.save(os.path.join(output_phy, 'similar_templates.npy'), similar_templates_new)

# For feature space (PCA)
# ====================================================================
# ====================================================================
# print("========================Starting PCA======================")
# DEBUG CHEK check after done
# mpath = os.path.join(output_phy,'pc_features.npy')
# x = np.load(mpath, allow_pickle=True).squeeze()
# # x = np.memmap(mpath)
# x1 = x[:, 1]
# x1 = x1[~np.isnan(x1)]
# x2 = x[:, 0]
# x2 = x2[~np.isnan(x2)]
# print(x.shape)
# # x = x[:6000, :, :].squeeze()
# # xxxx = []
# # for k in range(template_waveforms_new.shape[0]):
# #     x1 = x[np.where((spike_clusters_new==k) & (spike_times_new<1000))[0], :]
# #     xxxx.append(x1)
# # plt.figure()
# # for xxx in xxxx:
# #     plt.scatter(xxx[:, 1], xxx[:, 0], s=1)
# # plt.scatter(x1[:, 1], x1[:, 0])
# # plt.scatter(x2[:, 1], x2[:, 0])

# # plt.xlim(np.mean(x1)-np.std(x1), np.mean(x1)+np.std(x1))
# # plt.ylim(np.mean(x2)-np.std(x2), np.mean(x2)+np.std(x2))
# plt.scatter(x[:500, 1], x[:500, 0], s=1)
# plt.show()
# ###END DEBUG CHEK


# import multiprocessing
# from utils.pca_utils import compute_pca_by_channel, get_nearest_neighboors, compute_pca_single_channel
# N_JOBS = 32
# # from tqdm import tqdm
# nSpikes_curated = spike_clusters_new.shape[0]
# n_units_curated = template_waveforms_new.shape[0]
# prim_ch_by_unit_b0 = np.full(n_units_curated, -1)
# for spk_label, pri_ch in zip(spike_clusters_new, prim_ch_new):
#     prim_ch_by_unit_b0[spk_label] = pri_ch-1

# prim_locations = channel_positions[prim_ch_by_unit_b0, :] # (n_untis, 2)
# print(n_units_curated)
# # exit(0)
# # for now we only do k_chs=1 because ffs we did not record the non-primary-channel waveforms at each spike occurences
# # so this should equal prim_locations
# # TODO use the more reliable way of `getting cluster_mapping`
# MULTI_CHANNEL = False
# ts = time()
# neighbor_channels = get_nearest_neighboors(prim_locations, channel_positions, k_chs=1) # (n_units, 1)
# print("Time for neighboorhoold channel determiniation: %.2f" % (time()-ts))

# len_waveform = template_waveforms_new.shape[1]
# n_chs = template_waveforms_new.shape[2]
# temp_wavs_fname = os.path.join(output_phy, "all_waveforms_temp.npy")
# if os.path.exists(temp_wavs_fname):
#     os.unlink(temp_wavs_fname)


# if MULTI_CHANNEL:
#     all_waveforms = np.memmap(temp_wavs_fname, shape=(nSpikes_curated, len_waveform, n_chs), dtype=all_waveforms_by_cluster["clus1"].dtype, mode="w+")
# else:
#     all_waveforms = np.memmap(temp_wavs_fname, shape=(nSpikes_curated, len_waveform), dtype=all_waveforms_by_cluster["clus1"].dtype, mode="w+")


# def helper_prepare_one_unit(i_clus):
#     # print(i_clus, end=" ")
#     i_clus_original = int(cluster_mapping[i_clus, 1])
#     spk_ids_global = np.where(spike_clusters_new==i_clus)[0]
#     stamp_this_cluster = spike_times_new[spk_ids_global]
#     primch_this_cluster = prim_ch_by_unit_b0[i_clus]
#     waveforms_this_cluster = all_waveforms_by_cluster['clus%d'%(i_clus_original+1)]  # cluster IDs in .npz start from 1; (n_spikes, len_waveforms)
#     # print(waveforms_this_cluster.shape)
#     # if the first spike happens before we can capture the complete "pre-peak" part of the transient waveform, 
#     # then we don't include that spike in the PCA
#     spk_ids_local_maskfullwav = (stamp_this_cluster>=int((len_waveform-1)/2))
#     spk_ids_global_withfullwav = spk_ids_global[spk_ids_local_maskfullwav] 
#     # print(spk_ids_global_withfullwav.shape)
#     wvfm_valid_prim = waveforms_this_cluster#[spk_ids_local_maskfullwav, :] # (n_spikes, len_waveform)
#     # print(wvfm_valid_prim.shape)
#     if wvfm_valid_prim.shape[0] == spk_ids_global_withfullwav.shape[0]:
#         # print(all_waveforms[stamp_this_cluster, :, primch_this_cluster].shape)
#         if MULTI_CHANNEL:
#             all_waveforms[spk_ids_global_withfullwav, :, primch_this_cluster] = wvfm_valid_prim
#         else:
#             all_waveforms[spk_ids_global_withfullwav, :] = wvfm_valid_prim
#     elif wvfm_valid_prim.shape[0] == stamp_this_cluster.shape[0]-1:
#         if MULTI_CHANNEL:
#             all_waveforms[spk_ids_global_withfullwav[:-1], :, primch_this_cluster] = wvfm_valid_prim # ditto for the last spike
#         else:
#             all_waveforms[spk_ids_global_withfullwav[:-1], :] = wvfm_valid_prim
#     else:
#         raise ValueError("#waveforms and #spikes do not match. Likely that more than 2 spike occurences are missing")

# def helper_prepare_units(i_beg, i_end):
#     for i_u in range(i_beg, i_end):
#         if i_u >= n_units_curated:
#             break
#         print (i_u)
#         helper_prepare_one_unit(i_u)
#     all_waveforms.flush()

# n_units_per_job = int(np.ceil(n_units_curated/N_JOBS))
# print(n_units_per_job)
# procs = []
# for i_job in range(N_JOBS):
#     procs.append(multiprocessing.Process(target=helper_prepare_units, args=(i_job*n_units_per_job, (i_job+1)*n_units_per_job)))
# for proc in procs:
#     proc.start()
# for proc in procs:
#     proc.join()
# helper_prepare_units(0, n_units_curated)

# all_waveforms.flush()
# print("Saved temp waveforms..")
# # exit(0)
# mpath = os.path.join(output_phy,'pc_features.npy')
# mpatht = os.path.join(output_phy,'pc_features_t.npy')
# ts = time()
# if MULTI_CHANNEL:
#     pcs = compute_pca_by_channel(all_waveforms, spike_clusters_new, neighbor_channels, mpatht)
# else:
#     pcs = compute_pca_single_channel(all_waveforms, spike_clusters_new, prim_ch_by_unit_b0, mpatht)
# print(pcs.shape)
# print("Time for PCA: %.2f"% (time()-ts))
# np.save(mpath, pcs)
# np.save(os.path.join(output_phy, "pc_feature_ind.npy"), neighbor_channels)
# os.unlink(temp_wavs_fname)

# mpath = os.path.join(output_phy,'pc_features.npy')
# x = np.load(mpath, allow_pickle=True)
# # x = np.memmap(mpath)
# print(x.shape)
# x = x[:2000, :, :].squeeze()
# x1 = x[np.where((spike_clusters_new==1) & (spike_times_new<5000))[0], :]
# x2 = x[np.where((spike_clusters_new==3) & (spike_times_new<5000))[0], :]
# plt.figure()
# plt.scatter(x[:, 1], x[:, 0], s=0.2)
# # plt.scatter(x1[:, 1], x1[:, 0])
# # plt.scatter(x2[:, 1], x2[:, 0])
# plt.show()

# Convert back to MS outputs
def convert2MS(session_folder):
    firings_filepath = os.path.join(session_folder,"firings_clean_merged.mda")
    templates_filepath = os.path.join(session_folder,"templates_clean_merged.mda")
    
    # Re-creating firings.mda and templates.mda
    spike_clusters = np.load(os.path.join(session_folder,'phy_output','spike_clusters.npy'))
    spike_times = np.load(os.path.join(session_folder,'phy_output','spike_times.npy'))
    template_waveforms = readmda(os.path.join(session_folder, "templates_clean.mda")).astype(np.float64)
    cluster_info_filepath = os.path.join(session_folder,'phy_output','cluster_info.tsv')
    cluster_id_orig = np.arange(0,template_waveforms.shape[2],1)                    # original cluster IDs
    
    assert spike_times.shape == spike_clusters.shape, f"Array dimensions for spike_clusters.npy are not the same as spike_times.npy in session {session_folder}"
    
    if os.path.isfile(cluster_info_filepath):
        cluster_info_df = pd.read_csv(cluster_info_filepath,sep = '\t')
        cluster_label = cluster_info_df.group.to_numpy()
        cluster_id = cluster_info_df.cluster_id.to_numpy()
        n_spikes = cluster_info_df.n_spikes.to_numpy()
    else:
        shutil.copy(os.path.join(session_folder,'firings_clean.mda'),os.path.join(session_folder,'firings_clean_merged.mda'))
        shutil.copy(os.path.join(session_folder,'templates_clean.mda'),os.path.join(session_folder,'templates_clean_merged.mda'))
        return None             # PHY curation not performed
    
    # Loading original firings_clean.mda
    firings_original = readmda(os.path.join(session_folder,'firings_clean.mda'))
    spike_clusterIDs_original = firings_original[2,:] - 1                                                   # firings.mda cluster IDs start from 1  (mountainsort convention)
    firings_original[2,:] = spike_clusterIDs_original

     # Get what clusters IDs were merged
    df_phy_merge_info = GetMergeIDs(firings_original[2,:] , spike_clusters )
    
    # recompute the primary channel here
    newIDs = df_phy_merge_info.newID.to_numpy()
    oldIDs = df_phy_merge_info.oldIDs.tolist()
    
    local_templates_newIDs = np.zeros([template_waveforms.shape[0],template_waveforms.shape[1], newIDs.shape[0]])
    
    for iter_l,data_l in enumerate(newIDs):
        local_oldID = np.asarray(oldIDs[iter_l],dtype = np.int16)

        local_nspikes = [(spike_clusterIDs_original == data_ll).sum() for data_ll in local_oldID]        # number of spikes
        local_nspikes = np.asarray(local_nspikes,dtype = np.int32)
        
        weights_local = [local_nspikes[iter_ll]/local_nspikes.sum() for iter_ll in range(local_oldID.shape[0])]

        # weighted average of the templates_waveforms 
        local_templates_newIDs[:,:,iter_l] = np.average(template_waveforms[:,:,local_oldID],axis = 2,weights = weights_local )
        
    # New templates.mda    
    template_waveforms_new = np.zeros([template_waveforms.shape[0],template_waveforms.shape[1], cluster_id.shape[0]])
    for iter_l,data_l in enumerate(cluster_id):
        if np.any(np.isin(cluster_id_orig,data_l)):
            template_waveforms_new[:,:,iter_l] = template_waveforms[:,:,data_l]
            # print(data_l)
        else:
            indx_tmp = np.squeeze(np.where(newIDs == data_l))
            template_waveforms_new[:,:,iter_l] = local_templates_newIDs[:,:,indx_tmp]
            # print(indx_tmp)
    
    
    # Rejecting noise clusters here
    noise_clusters = cluster_id[(cluster_label == 'noise')]                                                      # These are the noise clusters (from PHY curation)
    if noise_clusters.any():
        spike_clusters_new = np.delete( spike_clusters , np.isin(spike_clusters, noise_clusters) , axis = 0 )        # Removing Manually curated units
        spike_times_new = np.delete( spike_times , np.isin(spike_clusters, noise_clusters) , axis = 0 )              # Removing Manually curated units
        template_waveforms_new = np.delete(template_waveforms_new,noise_clusters,axis = 2)                                   # Removing Manually curated units
        # cluster_id_orig = np.delete(cluster_id_orig,noise_clusters,axis = 0)    
        cluster_id = cluster_id[np.logical_not(cluster_label == 'noise')]            # manually curated
        n_spikes = n_spikes[np.logical_not(cluster_label == 'noise')]                # manually curated
        # firings_original = np.delete(firings_original, np.isin(firings_original[2,:] , noise_clusters), axis = 1)     # Removing Manually curated units    
    else:
        spike_clusters_new = spike_clusters
        spike_times_new = spike_times
        template_waveforms_new = template_waveforms_new
        
    assert template_waveforms_new.shape[2] == cluster_id.shape[0]
    
    # Primary channels (updated)
    pri_ch_LUT = -1 * np.ones(np.shape(cluster_id))
    pri_ch_new = -1 * np.ones(np.shape(spike_times_new))
    for iter_l in range(cluster_id.shape[0]):
        local_template = template_waveforms_new[:,:,iter_l] 
        local_template = -1*np.abs(-1*local_template)       # only care about the negative part of the spike 
        local_min_template = np.min(local_template[:,20:80],axis = 1)   # excluding the edges of the template waveform (in time)
        pri_ch_LUT[iter_l] = np.argmin(local_min_template)        # primary channel (electrode that shows highest depolarization voltage)
    for iter_l,data_l in enumerate(cluster_id): 
        tmp_indx_mask = (spike_clusters_new == data_l)
        local_pri_ch = pri_ch_LUT[iter_l]
        pri_ch_new[tmp_indx_mask] = local_pri_ch
    pri_ch_new += 1                                               # firings.mda channel count starts from 1
    spike_clusters_new += 1                                       # firings.mda channel count starts from 1
    
    # Generating new cluster IDs
    cluster_id_new_final = np.arange(1,cluster_id.shape[0]+1,1)
    cluster_id += 1
    spike_clusters_new_final = np.zeros(np.shape(spike_clusters_new),dtype = np.int16)
    for iter_l,data_l in enumerate(cluster_id):
        tmp_indx_mask = (spike_clusters_new == data_l)
        local_chan_id = cluster_id_new_final[iter_l]
        spike_clusters_new_final[tmp_indx_mask] = local_chan_id
        
    # Writing to disk
    firings = np.zeros([3,spike_times_new.shape[0]])
    firings[1,:] = spike_times_new
    firings[2,:] = spike_clusters_new_final
    firings[0,:] = pri_ch_new
    writemda64( firings, firings_filepath )
    writemda64(template_waveforms_new, templates_filepath)
    
    # Check primary channels IDs (native channel order OR MS channel order) for the convention in firings.mda (ask Jiaao)
    
def GetMergeIDs(spike_clusterIDs_original,spike_clusterIDs_new):
    # INPUT PARAMETERS
    # spike_clusterIDs_original : Array of the cluster IDs at all the spike stamps before PHY manual curation
    # spike_clusterIDs_new : Array of the cluster IDs at all the spike stamps after PHY manual curation 
    
    clusters_orig = np.unique(spike_clusterIDs_original)
    clusters_new = np.unique(spike_clusterIDs_new)
    
    clus_thresh = np.amax(clusters_orig)
    
    indx_phy = (spike_clusterIDs_new > clus_thresh)
    
    merged_clusters = clusters_new[clusters_new > clus_thresh]
    
    output_list = []    # create 2d list 
    
    df_output = pd.DataFrame(
            {
                "newID": [],
                "oldIDs": []
            }
            )
    
    for iter_c in merged_clusters:
        tmp_indx = np.where(spike_clusterIDs_new == iter_c)
        iter_c_orig = np.unique(spike_clusterIDs_original[tmp_indx])
        pd_tmp = pd.DataFrame({'newID':iter_c , 'oldIDs': [iter_c_orig.tolist()] })
        df_output = pd.concat((df_output,pd_tmp),axis = 0)                              # contains info for the merged cluster ID and the original cluster IDs
        
    return df_output
        
        
    
    
    
    
    
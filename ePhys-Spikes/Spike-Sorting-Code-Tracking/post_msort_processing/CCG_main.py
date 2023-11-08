import os
from time import time
from copy import deepcopy
from pycorrelate import pcorrelate
import gc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

from utils.read_mda import readmda

# Important parameters to control
F_SAMPLE = 25000
TRANSIENT_AMPLITUDE_VALID_DURATION = 1.5e-3 # seconds (duration of data before and after each spike that we consider when deciding the transient amplitude)
CCG_DURATION = 0.120 # in s, sum of both sides
CCG_BINSIZE  = 0.0004 # in s, smallets time resolution to count spikes

# Start of code: Initialization
TAVD_NSAMPLE = int(np.ceil(TRANSIENT_AMPLITUDE_VALID_DURATION*F_SAMPLE))
ccg_nbins = int(np.ceil(CCG_DURATION/CCG_BINSIZE))
ccg_bins = F_SAMPLE * np.linspace(-CCG_DURATION/2, CCG_DURATION/2, num=ccg_nbins+1)     # in samples

# Input data location (output of MS + curation)
session_folder1 = "/home/hyr2/Documents/Data/BC7/temp-MS-parameter-TEST/12-09-2021-param2/"
out_dir_ccg = os.path.join(session_folder1,'CCG_plots')
out_dir_templates = os.path.join(session_folder1,'templates')
os.makedirs(out_dir_ccg, exist_ok=True)

def get_pairwise_L2_distance_square_matrix(data_a, data_b):
    """
    assumes data_a is (n_1, n_features) and data_b is (n_2, n_features)
    returns (n_1, n_2) distance matrix
    where n_1 and n_2 could be cluster counts from 2 sessions
    """
    return np.sum((data_a[:,None,:]-data_b[None,:,:])**2, axis=2) / data_a.shape[1]

def get_pairwise_L2_distance_square_matrix_normed(data_a, data_b):
    """
    assumes data_a is (n_1, n_features) and data_b is (n_2, n_features)
    returns (n_1, n_2) distance matrix
    where n_1 and n_2 could be cluster counts from 2 sessions
    """
    norm_a = np.sum(data_a**2, axis=1) # (n_1,)
    norm_b = np.sum(data_b**2, axis=1) # (n_2,)
    norm_min = np.minimum(norm_a[:,None], norm_b[None,:]) # (n_1, n_2)
    return np.sum((data_a[:,None,:]-data_b[None,:,:])**2, axis=2) / norm_min

firings = readmda(os.path.join(session_folder1, "firings.mda")).astype(np.int64)
templates = readmda(os.path.join(session_folder1, "templates.mda"))
n_ch, n_sample, n_clus = templates.shape
masks = np.load(os.path.join(session_folder1, "cluster_rejection_mask.npz"))
mask = np.logical_or(masks['single_unit_mask'], masks['multi_unit_mask'])
my_slice = slice(int(n_sample//2-TAVD_NSAMPLE), int(n_sample//2+TAVD_NSAMPLE), 1)
templates = templates[:,my_slice,:]
n_sample_sliced = templates.shape[1]                                                    # # of samples in the sliced template matrix

# Read spike times and separate by clusters:
df = pd.DataFrame(firings)
df_spikes_by_clusters = pd.DataFrame(columns=['prim_chan', 'spike_stamps','cell_id'])
for iter_local in range(1,n_clus+1,1):
    # extracting time stamps of each cluster
    mask_bin_cluster = (df.iloc[2,:].values == iter_local)
    temp_indx = mask_bin_cluster.nonzero()[0][0]
    df_spikes_by_clusters.at[iter_local,'prim_chan'] =  df.iloc[0,temp_indx]
    df_spikes_by_clusters.at[iter_local,'cell_id'] =  df.iloc[2,temp_indx]
    mask_bin_cluster = df.iloc[1,:][mask_bin_cluster]
    mask_bin_cluster = mask_bin_cluster.to_numpy()
    df_spikes_by_clusters.at[iter_local,'spike_stamps'] = mask_bin_cluster
    
# Deleting temporary dataframe
del [[df]]
gc.collect()
df=pd.DataFrame()

# Loading the merge mask based on "location of nearby clusters" and their "waveform shape":
file_merge_csv = os.path.join(session_folder1,'merge_mask.csv')
df_merge_cluster = pd.read_csv(file_merge_csv,',',header = None, index_col = False)
arr_merge_cluster = df_merge_cluster.to_numpy()
cluster_pairs = arr_merge_cluster[:,[0,1]]
distance_mask = arr_merge_cluster[:,2]
# Extracting unique clusters from the merge_mask.csv and computing their absolute FR
p1 = np.unique(cluster_pairs[:,0],return_index=True, return_counts=True)
p2 = np.unique(cluster_pairs[:,1],return_index=True, return_counts=True)
p3 = np.concatenate((p1[0],p2[0]),axis = 0)
p3 = np.unique(p3)
arr_FR = np.zeros([n_clus,])
for iter_local in range(1,n_clus+1,1):  # overall firing rate of this cluster for the entire experiment
    arr_FR[iter_local-1] = F_SAMPLE * df_spikes_by_clusters.loc[iter_local,'spike_stamps'].size/(df_spikes_by_clusters.loc[iter_local,'spike_stamps'][-1] - df_spikes_by_clusters.loc[iter_local,'spike_stamps'][0])

# Making sure we dont double count the clusters (ie 2 cant be merged with 5 if 5 is being merged with 7)
# Make sure sparse FR should be ith index while high FR cluster should be jth index
# clus_indx_i = 1
# clus_indx_j = 1
local_iter = 0
for (clus_indx_i, clus_indx_j) in zip(cluster_pairs[:,0],cluster_pairs[:,1]):
    if distance_mask[local_iter] == 1:
        # if (arr_FR[clus_indx_j-1] > arr_FR[clus_indx_i-1]) and (clus_indx_i != clus_indx_j):
        #     # Compute CCGs (non-normalized ie we are computing counts only instead of FR)
        #     G_pairwise = pcorrelate(df_spikes_by_clusters.loc[clus_indx_i,'spike_stamps'],df_spikes_by_clusters.loc[clus_indx_j,'spike_stamps'],ccg_bins, normalize = False)
        # elif ((arr_FR[clus_indx_i-1] > arr_FR[clus_indx_j-1]) and (clus_indx_i != clus_indx_j)):
        #     G_pairwise = pcorrelate(df_spikes_by_clusters.loc[clus_indx_j,'spike_stamps'],df_spikes_by_clusters.loc[clus_indx_i,'spike_stamps'],ccg_bins, normalize = False)
        G_pairwise = pcorrelate(df_spikes_by_clusters.loc[clus_indx_i,'spike_stamps'],df_spikes_by_clusters.loc[clus_indx_j,'spike_stamps'],ccg_bins, normalize = False)
        if clus_indx_i == clus_indx_j:
            G_pairwise[int(ccg_nbins/2)] = 0             # at 0 index the value in counts is the number of spikes in the that cluster (only for ACG) 
        
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(5, 4)
        ax1 = fig.add_subplot(gs[0:3, :])
        ax1.set_title('CCG plot: ' + str(clus_indx_i) + ' - ' + str(clus_indx_j))
        ax2 = fig.add_subplot(gs[3:5, 0:2])
        ax2.set_title('Cluster ID:' + str(clus_indx_i))
        ax3 = fig.add_subplot(gs[3:5, 2:4])
        ax3.set_title('Cluster ID:' + str(clus_indx_j))
        # fig, (ax1,ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'height_ratios': [1,1,1],'width_ratios': [2,1,1]})
        # fig = plt.figure()
        # ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan = 1, colspan = 3)
        # ax2 = plt.subplot2grid((3, 3), (0, 1), rowspan = 1, colspan = 1)
        # ax3 = plt.subplot2grid((3, 3), (1, 0), rowspan = 1, colspan = 1)
        # ax4 = plt.subplot2grid((3, 3), (1, 1), rowspan = 1, colspan = 1)
        ax1.bar(1000*0.5*(ccg_bins[1:]+ccg_bins[:-1])/F_SAMPLE,G_pairwise, align = 'center',color = 'b', edgecolor = 'k', width = 0.25)     # x axis in ms
        ax1.set_xlabel('Delay (ms)')
        ax1.set_ylabel('Counts')
        ax1.set_xlim(-30,30)
        # Plotting templates
        t_i = np.reshape(templates[df_spikes_by_clusters.loc[clus_indx_i,'prim_chan']-1,:,clus_indx_i-1],[n_sample_sliced,])
        t_j = np.reshape(templates[df_spikes_by_clusters.loc[clus_indx_j,'prim_chan']-1,:,clus_indx_j-1],[n_sample_sliced,])
        t_lim = n_sample_sliced/(2*F_SAMPLE)
        t_axis = np.linspace(-t_lim*1000,t_lim*1000,76)
        ax2.plot(t_axis,t_i)
        ax2.set_ylim([-240,60])
        ax2.set_ylabel('uV')
        ax2.set_xlabel('ms')
        ax3.plot(t_axis,t_j)
        ax3.set_ylim([-240,60])
        ax3.set_ylabel('uV')
        ax3.set_xlabel('ms')
        # Saving
        filename_save = str(clus_indx_i) + '-' + str(clus_indx_j) + '.svg'
        filename_save = os.path.join(out_dir_ccg,filename_save)
        fig.set_size_inches((9, 8), forward=True)
        plt.savefig(filename_save,format = 'svg')
        plt.close('all')
        plt.clf()
        plt.cla()
    local_iter += 1

# Computing ISIs (instead of ACGs since they are computationally easier). There is a minor difference between ISI and ACG. ISI looks at nearest neighbout only.
 # ------------------------------ ####

# template_features = template_features[mask,:]
# print(template_features.shape)
# ts = time()
# dist_mat = get_pairwise_L2_distance_square_matrix(template_features, template_features)
# print(time()-ts)
# plt.figure(); plt.imshow(dist_mat, cmap='seismic'); plt.colorbar(); plt.show()
# plt.figure(); plt.hist(dist_mat[np.triu_indices(dist_mat.shape[0],k=1)].ravel(), bins=50); plt.show()
# for i_ch in range(n_ch):
#     neighborhood_mask = np.logical_and((pri_ch_lut==i_ch), mask)
#     n_neighborhood = np.sum(neighborhood_mask)
#     if n_neighborhood<=1:
#         continue
#     neighborhood_clus_ids = np.where(neighborhood_mask)[0]
#     dist_mat = get_pairwise_L2_distance_square_matrix(template_features[neighborhood_mask,:], template_features[neighborhood_mask,:])
#     # dist_mat = np.corrcoef(template_features[neighborhood_mask, :])
#     plt.figure(); plt.imshow(dist_mat, cmap='gray'); plt.colorbar(); 
#     plt.xticks(np.arange(n_neighborhood), neighborhood_clus_ids+1)
#     plt.yticks(np.arange(n_neighborhood), neighborhood_clus_ids+1)
#     plt.show()

# n_ch = size(templates,1);
# n_clus = size(templates, 3);
# spike_times_all = firings(2,idx) ;
# disp(size(spike_times_all))
# spike_labels = firings(3,idx);
# ch_stamp = firings(1,idx);
# spike_times_by_clus = cell(1, n_clus);
# ts_by_clus = cell(1, n_clus);
# pri_ch_lut = -1*ones(1, n_clus);
# for i=1:n_clus
#     spike_times_by_clus{i} = [];
#     ts_by_clus{i} = [];
# end
# % count spikes by unit
# for i=1:length(spike_times_all)
#     spk_lbl = spike_labels(i);
#     pri_ch_lut(spk_lbl) = ch_stamp(spk_lbl);
#     spike_times_by_clus{spk_lbl}(end+1) = spike_times_all(i)/Fs;
#     ts_by_clus{spk_lbl}(end+1) = spike_times_all(i);
# end




# get primary channel for each label; safely assumes each cluster has only one primary channel
# pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
# n_pri_ch_known = 0
# for (spk_ch, spk_lbl) in zip(firings[0,:], firings[2,:]):
#     if pri_ch_lut[spk_lbl-1]==-1:
#         pri_ch_lut[spk_lbl-1] = spk_ch-1
#         n_pri_ch_known += 1
#         if n_pri_ch_known==n_clus:
#             break
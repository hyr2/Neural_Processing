#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:53:40 2024

@author: hyr2-office
"""

# script for generating Figure 2C
import os, json, time
import sys
sys.path.append('utils')
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy import integrate
from scipy.stats import ttest_ind, zscore, norm
from itertools import product
import pandas as pd
from utils.read_mda import readmda
from utils.read_stimtxt import read_stimtxt
import seaborn as sns
from natsort import natsorted
from matplotlib.cm import ScalarMappable

def func_pop_analysis(session_folder,CHANNEL_MAP_FPATH,output_folder):

    # Input parameters ---------------------
    # session folder is a single measurement 
    
    # firing rate calculation params
    WINDOW_LEN_IN_SEC = 40e-3
    SMOOTHING_SIZE = 11
    DURATION_OF_INTEREST = 2.5  # how many seconds to look at upon stim onset (this is the activation or inhibition window)
    # Setting up
    session_trialtimes = os.path.join(session_folder,'trials_times.mat')
    trial_mask_file = os.path.join(session_folder,'trial_mask.csv')
    result_folder = os.path.join(session_folder,'Processed', 'count_analysis')
    result_folder_FR_avg = os.path.join(session_folder,'Processed', 'FR_clusters')
    # dir_expsummary = os.path.join(session_folder,'exp_summary.xlsx')
    
    # Extract sampling frequency
    file_pre_ms = os.path.join(session_folder,'pre_MS.json')
    with open(file_pre_ms, 'r') as f:
      data_pre_ms = json.load(f)
    F_SAMPLE = float(data_pre_ms['SampleRate'])
    CHMAP2X16 = bool(data_pre_ms['ELECTRODE_2X16'])      # this affects how the plots are generated
    Num_chan = int(data_pre_ms['NumChannels'])
    Notch_freq = float(data_pre_ms['Notch filter'])
    Fs = float(data_pre_ms['SampleRate'])
    stim_start_time = float(data_pre_ms['StimulationStartTime'])
    n_stim_start = int(Fs * stim_start_time)
    Ntrials = int(data_pre_ms['NumTrials'])
    stim_end_time = stim_start_time + float(data_pre_ms['StimulationTime'])
    time_seq = float(data_pre_ms['SequenceTime'])
    Seq_perTrial = float(data_pre_ms['SeqPerTrial'])
    total_time = time_seq * Seq_perTrial
    print('Each sequence is: ', time_seq, 'sec')
    time_seq = int(np.ceil(time_seq * Fs/2))
   
    # read trials times
    trials_start_times = loadmat(session_trialtimes)['t_trial_start'].squeeze()

    # Trial mask (only for newer animals)
    if os.path.isfile(trial_mask_file):
        trial_mask = pd.read_csv(trial_mask_file, header=None, index_col=False,dtype = bool)
        TRIAL_KEEP_MASK = trial_mask.to_numpy(dtype = bool)
        TRIAL_KEEP_MASK = np.squeeze(TRIAL_KEEP_MASK)
    else:
        TRIAL_KEEP_MASK = np.ones([trials_start_times.shape[0],],dtype = bool)
    # TRIAL_KEEP_MASK = np.ones([Ntrials,],dtype = bool)

    # Channel mapping
    if (CHMAP2X16 == True):    # 2x16 channel map
        GH = 30
        GW_BWTWEENSHANKS = 250
    elif (CHMAP2X16 == False):  # 1x32 channel map
        GH = 25
        GW_BWTWEENSHANKS = 250

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(result_folder_FR_avg):
        os.makedirs(result_folder_FR_avg)
    geom_path = os.path.join(session_folder, "geom.csv")
    # curation_mask_path = os.path.join(session_folder, 'accept_mask.csv')        # deparcated
    NATIVE_ORDERS = np.load(os.path.join(session_folder, "native_ch_order.npy"))
    # axonal_mask = os.path.join(session_folder,'positive_mask.csv')              # deparcated

    # macro definitions
    ANALYSIS_NOCHANGE = 0       # A better name is non-stimulus locked
    ANALYSIS_EXCITATORY = 1     # A better name is activated
    ANALYSIS_INHIBITORY = -1    # A better name is suppressed

    # read cluster rejection data (# deparcated)
    # curation_masks = np.squeeze(pd.read_csv(curation_mask_path,header = None).to_numpy())   
    # single_unit_mask = curation_masks
    # single_unit_mask = curation_masks['single_unit_mask']
    # multi_unit_mask = curation_masks['multi_unit_mask']
    # mask_axonal = np.squeeze(pd.read_csv(axonal_mask,header = None).to_numpy())  # axonal spikes
    # mask_axonal = np.logical_not(mask_axonal)
    # single_unit_mask = np.logical_and(mask_axonal,single_unit_mask)     # updated single unit mask
    
    chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
    # print(list(chmap_mat.keys()))
    if np.min(chmap_mat)==1:
        print("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
        chmap_mat -= 1

    def ch_convert_msort2intan(i_msort):
        """i_msort starts from 0; return index also starts from zero"""
        # x, y = geom[i_msort,:]
        # i_intan = chmap_mat[int(y/GH), int(x/GW)] # the real index in the complete 128 channel map
        # return i_intan
        return NATIVE_ORDERS[i_msort]

    def ch_convert_intan2msort(i_intan):
        """i_intan starts from 0; return msort index also starts from zero"""
        # pos = np.where(chmap_mat==i_intan)
        # y_idx, x_idx = pos[0][0], pos[1][0]
        # i_msort = np.where(np.logical_and(geom[:,0]==int(x_idx*GW), geom[:,1]==int(y_idx*GH)))
        # return i_msort
        return np.where(NATIVE_ORDERS==i_intan)[0][0]

    def get_shanknum_from_intan_id(i_intan):
        """i_intan is the native channel used in the .mat channel map starts from 0"""
        if CHMAP2X16:
            return int(np.where(chmap_mat==i_intan)[1][0])//2
        else:
            return int(np.where(chmap_mat==i_intan)[1][0])

    def get_shanknum_from_msort_id(i_msort):
        return get_shanknum_from_intan_id(NATIVE_ORDERS[i_msort])


    def get_shanknum_from_coordinate(x, y=None):
        "get shank number from coordinate"
        if isinstance(x, int):
            return int(x/GW_BWTWEENSHANKS)
        elif isinstance(x, np.ndarray) and x.shape==(2,):
            return int(x[0]/GW_BWTWEENSHANKS)
        else:
            raise ValueError("wrong input")

    # Read cluster locations
    clus_loc = pd.read_csv(os.path.join(session_folder,'clus_locations_clean_merged.csv'),header = None)
    clus_loc = clus_loc.to_numpy()
    trial_duration_in_samples = int(total_time*F_SAMPLE)
    window_in_samples = int(WINDOW_LEN_IN_SEC*F_SAMPLE)
    # read channel map
    geom = pd.read_csv(geom_path, header=None).values
    n_ch_this_session = geom.shape[0]
    print(geom.shape)
    # exit(0)
    firings = readmda(os.path.join(session_folder, "firings_clean_merged.mda")).astype(np.int64)
    # get spike stamp for all clusters (in SAMPLEs not seconds)
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
    spike_labels_unique = np.unique(spike_labels)
    single_unit_mask = np.ones(spike_labels_unique.shape,dtype = bool)
    n_clus = np.max(spike_labels)
    print('Total number of clusters found: ',n_clus)
    spike_times_by_clus =[[] for i in range(n_clus)]
    spike_count_by_clus = np.zeros((n_clus,))
    for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
        spike_times_by_clus[spk_lbl-1].append(spk_time)
    for i in range(n_clus):
        spike_times_by_clus[i] = np.array(spike_times_by_clus[i])
        spike_count_by_clus[i] = spike_times_by_clus[i].shape[0]
    # get primary channel for each label
    pri_ch_lut = -1*np.ones(n_clus, dtype=int)
    tmp_cnt = 0
    for (spk_ch, spk_lbl) in zip(firings[0,:], spike_labels):
        if pri_ch_lut[spk_lbl-1]==-1:
            pri_ch_lut[spk_lbl-1] = spk_ch-1
            if tmp_cnt == n_clus-1:
                break
            else:
                tmp_cnt += 1


    def get_single_cluster_spikebincouts_all_trials(firing_stamp, t_trial_start, trial_duration_in_samples, window_in_samples):
        # returns the spike bin counts; need to manuaaly divide result by window length to get real firing rate
        n_windows_in_trial = int(np.ceil(trial_duration_in_samples/window_in_samples))
        bin_edges = np.arange(0, window_in_samples*n_windows_in_trial+1, step=window_in_samples)
        n_trials = t_trial_start.shape[0]
        # print(n_trials,Ntrials)
        assert n_trials==Ntrials or n_trials==Ntrials+1, "%d %d" % (n_trials, Ntrials)
        if n_trials > Ntrials:
            n_trials = Ntrials
        firing_rate_series_by_trial = np.zeros((n_trials, n_windows_in_trial))
        for i in range(n_trials):
            trial_firing_mask = np.logical_and(firing_stamp>=t_trial_start[i], firing_stamp<=(t_trial_start[i]+trial_duration_in_samples))
            trial_firing_stamp = firing_stamp[trial_firing_mask] - t_trial_start[i]
            if trial_firing_stamp.shape[0]==0:
                continue
            tmp_hist, _ = np.histogram(trial_firing_stamp, bin_edges)   # The firing rate series for each trial for a single cluster
            firing_rate_series_by_trial[i,:] = tmp_hist   
        return firing_rate_series_by_trial

    def single_cluster_main(i_clus):
        
        firing_stamp = spike_times_by_clus[i_clus] 
        N_spikes_local = spike_times_by_clus[i_clus].size
        
        stim_start_time =  n_stim_start     # number of samples after which stim begins
        stim_end_time = n_stim_start + (Fs * 1.5) # 1.5 s post stim start is set as the stim end time. We only consider the first 1500ms 
        bsl_start_time = Fs*0.5     # start of baseline period
        
        # Firing rate series
        firing_rate_series = get_single_cluster_spikebincouts_all_trials(
            firing_stamp, 
            trials_start_times, 
            trial_duration_in_samples, 
            window_in_samples
            )
        firing_rate_series = firing_rate_series[TRIAL_KEEP_MASK, :]/WINDOW_LEN_IN_SEC       # This gives firing rate in Hz
        firing_rate_avg = np.mean(firing_rate_series, axis=0) # averaging over all trials
        firing_rate_avg_nofilt = firing_rate_avg
        
        n_samples_baseline = int(np.ceil(stim_start_time/WINDOW_LEN_IN_SEC))
        n_samples_stim = int(np.ceil((stim_end_time-stim_start_time)/WINDOW_LEN_IN_SEC))
        
        t_axis = np.linspace(0,firing_rate_avg.shape[0]*WINDOW_LEN_IN_SEC,firing_rate_avg.shape[0])
        t_1 = np.squeeze(np.where(t_axis >= 1.4))[0]      # stim start set to 2.45 seconds
        t_2 = np.squeeze(np.where(t_axis >= 2.0))[0]      # end of bsl region (actual value is 2.5s but FR starts to increase before. Maybe anticipatory spiking due to training)
        t_3 = np.squeeze(np.where(t_axis <= 2.1))[-1]        # stim end set to 5.15 seconds
        t_4 = np.squeeze(np.where(t_axis <= 4.02))[-1]
        t_5 = np.squeeze(np.where(t_axis <= 8))[-1]        # post stim quiet period
        t_6 = np.squeeze(np.where(t_axis <= 9.5))[-1]
        t_7 = np.squeeze(np.where(t_axis <= 4.5))[-1]
        t_8 = np.squeeze(np.where(t_axis <= 2.520))[-1]       
        t_9 = np.squeeze(np.where(t_axis <= 0.6))[-1]       
    
        # Computing number of spikes
        firing_rate_series2 = firing_rate_series * WINDOW_LEN_IN_SEC    # Number of spikes (histogram over for all trials)
        firing_rate_series_avg = np.mean(firing_rate_series2,axis = 0)  # Avg number of spikes over accepted trials
        Spikes_stim = np.sum(firing_rate_series_avg[t_8:t_4])   # 1.5 sec
        Spikes_bsl = np.sum(firing_rate_series_avg[t_9:t_3])    # 1.5 sec
        Spikes_num = np.array([Spikes_bsl,Spikes_stim])

        return firing_rate_avg, Spikes_num


    total_nclus_by_shank = np.zeros(4)
    single_nclus_by_shank = np.zeros(4)
    multi_nclus_by_shank = np.zeros(4)
    act_nclus_by_shank = np.zeros(4)
    inh_nclus_by_shank = np.zeros(4)
    nor_nclus_by_shank = np.zeros(4)

    clus_response_mask = np.zeros(np.sum(single_unit_mask), dtype=int)

    FR_series_all_clusters = [ [] for i in range(np.sum(single_unit_mask)) ]   # create empty list 
    Avg_FR_byshank = np.zeros([4,])
    FR_list_byshank_act =  [ [] for i in range(4) ]  # create empty list
    FR_list_byshank_inh =  [ [] for i in range(4) ]  # create empty list
    list_all_clus = []
    iter_local = 0
    
    df_all_clust = pd.DataFrame(data=None,columns = ['cluster_id','shank_num','depth','N_spikes','spont_FR'])

    for i_clus in range(n_clus):
        
        if single_unit_mask[i_clus] == True:
            # Need to add facilitating cell vs adapting cell vs no change cell
            firing_rate_avg, Spikes_num = single_cluster_main(i_clus)
            t_axis = np.linspace(0,total_time,firing_rate_avg.shape[0])
            t_1 = np.squeeze(np.where(t_axis >= 8.5))[0]
            t_2 = np.squeeze(np.where(t_axis >= 12.5))[0]
            t_3 = np.squeeze(np.where(t_axis >= 2.35))[0]
            t_4 = np.squeeze(np.where(t_axis >= 3.75))[0]
            t_5 = 2.45
            
            t_bsl_start = np.squeeze(np.where(t_axis >= 0.25))[0]
            t_bsl_end = np.squeeze(np.where(t_axis >= 2.25))[0]
            # creating a dictionary for this cluster
            firing_stamp = spike_times_by_clus[i_clus]
            N_spikes_local = spike_times_by_clus[i_clus].size 
            shank_num = int(clus_loc[i_clus,0] // 250)      # starts from 0
            depth = int(clus_loc[i_clus,1])  # triangulation by Jiaao
            # shank_num = get_shanknum_from_msort_id(prim_ch)
            data_input = [i_clus + 1,shank_num,depth,N_spikes_local,np.mean(firing_rate_avg[t_bsl_start:t_bsl_end])]
            df_all_clust.loc[i_clus] = data_input
            
            # i_clus_dict  = {}
            # i_clus_dict['cluster_id'] = i_clus + 1 # to align with the discard_noise_viz.py code cluster order [folder: figs_allclus_waveforms]
            # i_clus_dict['shank_num'] = shank_num
            # i_clus_dict['depth'] = depth
            # i_clus_dict['N_spikes'] = N_spikes_local
            # i_clus_dict['spont_FR'] = np.mean(firing_rate_avg[t_bsl_start:t_bsl_end])
                           
            # Append to output dataframe of all units in current session
            # list_all_clus.append(i_clus_dict)

            iter_local = iter_local+1
    
    # Save the dataframe in output folder
    df_all_clust.to_pickle(os.path.join(output_folder,'all_clusters_.pkl'))
    
if __name__ == '__main__':
    
    # Work on RH7 
    # CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_flex_Pavlo.mat'
    
    # input_dir = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh7/'
    # source_dir_list = natsorted(os.listdir(input_dir))
    # session_ids = [-3,-2,2,7,14,21,28,35,42,49,56]
    
    # # Generating pkl files 
    # for Raw_dir in source_dir_list:
    #     Raw_dir_dir = os.path.join(input_dir,Raw_dir)
    #     if os.path.isdir(Raw_dir_dir):
    #         output_folder = os.path.join(Raw_dir_dir,'special_folder_Fig2')
            
    #         os.makedirs(output_folder)
    #         func_pop_analysis(Raw_dir_dir,CHANNEL_MAP_FPATH,output_folder)
            
    # # Reading pkl files and generating plots
    # source_dir_list = natsorted(os.listdir(input_dir))
    # df_all_clust = pd.DataFrame(data=None,columns = ['cluster_id','shank_num','depth','N_spikes','spont_FR','session'])
    # # Generating pkl files 
    # iter_iter = 0
    # dfs = []
    # for Raw_dir in source_dir_list:
    #     Raw_dir_dir = os.path.join(input_dir,Raw_dir)
    #     if os.path.isdir(Raw_dir_dir):
    #         output_folder = os.path.join(Raw_dir_dir,'special_folder_Fig2','all_clusters_.pkl')
    #         df_all_clust = pd.read_pickle(output_folder)
    #         df_all_clust['session'] = [session_ids[iter_iter]] * df_all_clust.shape[0]
    #         dfs.append(df_all_clust)
    #         iter_iter += 1
    # df_combined_rh7 = pd.concat(dfs)
    # df_combined_rh7.to_pickle(os.path.join(input_dir,'df_Fig2C.pkl'))
    
    # Work on BC7 
    # CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_flex_new.mat'
    
    # input_dir = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_bc7/'
    # source_dir_list = natsorted(os.listdir(input_dir))
    # session_ids = [-3,-2,2,7,14,21,28,42]
    
    # Generating pkl files 
    # for Raw_dir in source_dir_list:
    #     Raw_dir_dir = os.path.join(input_dir,Raw_dir)
    #     if os.path.isdir(Raw_dir_dir):
    #         output_folder = os.path.join(Raw_dir_dir,'special_folder_Fig2')
            
    #         os.makedirs(output_folder)
    #         func_pop_analysis(Raw_dir_dir,CHANNEL_MAP_FPATH,output_folder)
            
    # Reading pkl files and generating plots
    # source_dir_list = natsorted(os.listdir(input_dir))
    # df_all_clust = pd.DataFrame(data=None,columns = ['cluster_id','shank_num','depth','N_spikes','spont_FR','session'])
    # # Generating pkl files 
    # iter_iter = 0
    # dfs = []
    # for Raw_dir in source_dir_list:
    #     Raw_dir_dir = os.path.join(input_dir,Raw_dir)
    #     if os.path.isdir(Raw_dir_dir):
    #         output_folder = os.path.join(Raw_dir_dir,'special_folder_Fig2','all_clusters_.pkl')
    #         df_all_clust = pd.read_pickle(output_folder)
    #         df_all_clust['session'] = [session_ids[iter_iter]] * df_all_clust.shape[0]
    #         dfs.append(df_all_clust)
    #         iter_iter += 1
    # df_combined_bc7 = pd.concat(dfs)
    # df_combined_bc7.to_pickle(os.path.join(input_dir,'df_Fig2C.pkl'))
    
    # Work on RH8
    # CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_flex_Pavlo.mat'
    
    # input_dir = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh8/'
    # source_dir_list = natsorted(os.listdir(input_dir))
    # session_ids = [-3,-2,2,7,14,21,28,35,42,49,56]
    
    # # Generating pkl files 
    # for Raw_dir in source_dir_list:
    #     Raw_dir_dir = os.path.join(input_dir,Raw_dir)
    #     if os.path.isdir(Raw_dir_dir):
    #         output_folder = os.path.join(Raw_dir_dir,'special_folder_Fig2')
            
    #         os.makedirs(output_folder)
    #         func_pop_analysis(Raw_dir_dir,CHANNEL_MAP_FPATH,output_folder)
            
    # # Reading pkl files and generating plots
    # source_dir_list = natsorted(os.listdir(input_dir))
    # df_all_clust = pd.DataFrame(data=None,columns = ['cluster_id','shank_num','depth','N_spikes','spont_FR','session'])
    # # Generating pkl files 
    # iter_iter = 0
    # dfs = []
    # for Raw_dir in source_dir_list:
    #     Raw_dir_dir = os.path.join(input_dir,Raw_dir)
    #     if os.path.isdir(Raw_dir_dir):
    #         output_folder = os.path.join(Raw_dir_dir,'special_folder_Fig2','all_clusters_.pkl')
    #         df_all_clust = pd.read_pickle(output_folder)
    #         df_all_clust['session'] = [session_ids[iter_iter]] * df_all_clust.shape[0]
    #         dfs.append(df_all_clust)
    #         iter_iter += 1
    # df_combined_rh8 = pd.concat(dfs)
    # df_combined_rh8.to_pickle(os.path.join(input_dir,'df_Fig2C.pkl'))
    
    
    # OUTPUT PLOTTING
    
    t = time.localtime()
    current_time = time.strftime("%m_%d_%Y_%H_%M", t)
    current_time = current_time + '_Fig2C'
    # output_folder = os.path.join(r'C:\Rice_2023\Data\Results',current_time)
    output_folder = os.path.join('/home/hyr2-office/Documents/Data/NVC/Results',current_time)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    folder_bc7 = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_bc7/df_Fig2C.pkl'
    folder_rh8 = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh8/df_Fig2C.pkl'
    folder_rh7 = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_rh7/df_Fig2C.pkl'
    
    # Reading files BC7
    fig,axes = plt.subplots(1,1,figsize = (3,1.5),dpi = 300)
    df_all_clust_bc7 = pd.read_pickle(folder_bc7)
    df_all_clust_bc7['depth'] = 800 - df_all_clust_bc7['depth']     # depth correction for plotting
    df_all_clust_bc7 = df_all_clust_bc7.loc[(df_all_clust_bc7['shank_num'] == 3)]
    # df_bsl = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == -3) & (df_all_clust_bc7['shank_num'] == 0)]
    df_all_clust_bc7 = df_all_clust_bc7.loc[(df_all_clust_bc7['spont_FR'] > 0.5)]   # rejected sparse units
    # df_sa = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 2) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_rec = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 14) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_chr = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 42) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_all = pd.concat((df_bsl,df_sa,df_rec,df_chr),axis=0)
    cmap_1 = sns.color_palette("Spectral", as_cmap=True).reversed()
    sns.stripplot(data=df_all_clust_bc7,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes)
    sns.despine(left = True)
    axes.set_yticks([])
    axes.legend([],[], frameon=False)
    
    
    # For Figure 4: Beyond Peri Infarct
    # Selected BC7 shankD
    fig,axes = plt.subplots(1,1,figsize = (3,1.5),dpi = 300)
    df_all_clust_rh7 = pd.read_pickle(folder_bc7)
    df_all_clust_rh7['depth'] = 800 - df_all_clust_rh7['depth']     # depth correction for plotting
    df_all_clust_rh7 = df_all_clust_rh7.loc[(df_all_clust_rh7['shank_num'] == 3)]
    # df_bsl = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == -3) & (df_all_clust_bc7['shank_num'] == 0)]
    df_all_clust_rh7 = df_all_clust_rh7.loc[(df_all_clust_rh7['spont_FR'] > 1.25)]   # rejected sparse units
    # df_sa = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 2) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_rec = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 14) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_chr = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 42) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_all = pd.concat((df_bsl,df_sa,df_rec,df_chr),axis=0)
    cmap_1 = sns.color_palette("Spectral", as_cmap=True).reversed()
    df_all_clust_rh7_bsl = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == -3)]
    df_all_clust_rh7_2 = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == 2)]
    df_all_clust_rh7_7 = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == 7)]
    df_all_clust_rh7_21 = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == 21)]
    df_all_clust_rh7_56 = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == 42)]
    
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh7_bsl,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_bsl.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh7_2,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_2.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh7_7,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_7.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh7_21,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_21.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    # df_all_clust_rh8_56 = df_all_clust_rh7_56.loc[(df_all_clust_rh7_56['spont_FR'] > 1)]
    sns.stripplot(data=df_all_clust_rh7_56,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_56.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    
    
    
    # df_all_clust_rh8 = df_all_clust_rh8.filter(items = ['N_spikes','session'])
    # sz = df_all_clust_rh8.groupby('session').sum()
    # sz = np.squeeze(sz.to_numpy())
    # sz = 1000*sz / np.mean(sz[0:2])
    # x = df_all_clust_rh8['session'].unique()
    # y = np.ones(x.shape)
    # plt.scatter(x, y , s = sz, alpha = 1, c = '#4d4e4d')
    
    # RH7
    
    # Reading files RH7
    fig,axes = plt.subplots(1,1,figsize = (3,1.5),dpi = 300)
    df_all_clust_rh7 = pd.read_pickle(folder_rh7)
    df_all_clust_rh7['depth'] = 800 - df_all_clust_rh7['depth']     # depth correction for plotting
    df_all_clust_rh7 = df_all_clust_rh7.loc[(df_all_clust_rh7['shank_num'] == 1)]
    # df_bsl = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == -3) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_bsl = df_bsl.loc[(df_bsl['spont_FR'] > 1)]   # rejected sparse units
    # df_sa = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 2) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_rec = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 14) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_chr = df_all_clust_bc7.loc[(df_all_clust_bc7['session'] == 42) & (df_all_clust_bc7['shank_num'] == 0)]
    # df_all = pd.concat((df_bsl,df_sa,df_rec,df_chr),axis=0)
    cmap_1 = sns.color_palette("Spectral", as_cmap=True).reversed()
    df_all_clust_rh7_bsl = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == -3)]
    df_all_clust_rh7_7 = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == 14)]
    df_all_clust_rh7_21 = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == 28)]
    df_all_clust_rh7_56 = df_all_clust_rh7.loc[(df_all_clust_rh7['session'] == 42)]
    
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh7_bsl,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_bsl.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh7_7,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_7.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh7_21,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_21.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    # df_all_clust_rh8_56 = df_all_clust_rh7_56.loc[(df_all_clust_rh7_56['spont_FR'] > 1)]
    sns.stripplot(data=df_all_clust_rh7_56,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH7_shankA_Stroke_Spont_FR_56.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    
    # RH8
    
    # Reading files RH8
    df_all_clust_rh8 = pd.read_pickle(folder_rh8)
    df_all_clust_rh8['depth'] = 800 - df_all_clust_rh8['depth']     # depth correction for plotting
    df_all_clust_rh8 = df_all_clust_rh8.loc[(df_all_clust_rh8['shank_num'] == 0)]
    # df_bsl = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == -3) & (df_all_clust_rh8['shank_num'] == 0)]
    # df_bsl = df_bsl.loc[(df_bsl['spont_FR'] > 1)]   # rejected sparse units
    # df_sa = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == 2) & (df_all_clust_rh8['shank_num'] == 0)]
    # df_rec = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == 7) & (df_all_clust_rh8['shank_num'] == 0)]
    # df_chr = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == 49) & (df_all_clust_rh8['shank_num'] == 0)]
    # df_all = pd.concat((df_bsl,df_sa,df_rec,df_chr),axis=0)
    cmap_1 = sns.color_palette("Spectral", as_cmap=True).reversed()
    # df_all_clust_rh8 = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == -3) | (df_all_clust_rh8['session'] == 2) | (df_all_clust_rh8['session'] == 7) | (df_all_clust_rh8['session'] == 21) | (df_all_clust_rh8['session'] == 56)]
    df_all_clust_rh8_bsl = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == -3)]
    df_all_clust_rh8_7 = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == 7)]
    df_all_clust_rh8_21 = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == 21)]
    df_all_clust_rh8_56 = df_all_clust_rh8.loc[(df_all_clust_rh8['session'] == 56)]
    # plt.colorbar(label="Y Values")
    
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh8_bsl,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH8_shankA_Stroke_Spont_FR_bsl.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh8_7,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH8_shankA_Stroke_Spont_FR_7.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    sns.stripplot(data=df_all_clust_rh8_21,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH8_shankA_Stroke_Spont_FR_21.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    fig,axes = plt.subplots(1,1,figsize = (1,4.5),dpi = 300)
    df_all_clust_rh8_56 = df_all_clust_rh8_56.loc[(df_all_clust_rh8_56['spont_FR'] > 1)]
    sns.stripplot(data=df_all_clust_rh8_56,x='session',y='depth',hue='spont_FR',palette=cmap_1,ax = axes,jitter = 0.25,size = 11,edgecolor = 'k',linewidth=1,alpha = 0.7)
    sns.despine(top = True, right = True, bottom = True)
    axes.legend([],[], frameon=False)
    axes.set_xticks([])
    axes.set_yticks([800,400,0])
    # norm = plt.Normalize(0.5,20)
    # sm =  ScalarMappable(norm=norm, cmap=cmap_1)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes)
    # cbar.ax.set_title("scale")
    # cbar.ax.set_title('')
    fig.set_size_inches(1,4.5)
    filename_save = os.path.join(output_folder,'RH8_shankA_Stroke_Spont_FR_56.svg')
    plt.savefig(filename_save,format = 'svg',dpi = 500,transparent=True)
    
    
    
    
    
import os, json
import sys
sys.path.append('utils')
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy import integrate
from scipy.stats import ttest_ind, zscore, norm
from scipy.signal import savgol_filter
from itertools import product, compress
import pandas as pd
from utils.read_mda import readmda
from utils.read_stimtxt import read_stimtxt
from Support import plot_all_trials, filterSignal_lowpass, filter_Savitzky_slow, filter_Savitzky_fast, zscore_bsl
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns
import matlab.engine

def extract_waveforms_and_amplitude(filt_mda_single_channel,spiketimes_all_sessions,Fs,waveform_len=4e-3):
    # filt_mda_single_channel: the raw (filtered) electrophysiology data from a single electrode (the primary channel for the unit in question)
    # spiketimes_all_sessions: the spike times in samples of the unit organized by session
    # Fs: Sampling rate of the electrophysiological data
    # waveform_len: 4ms by default but can be changed
    spk_amp_series = []
    waveforms_all = [] # only store the real-time waveforms at primary channel for each cluster
    proper_spike_times_by_clus = []
    
    keys_ = spiketimes_all_sessions.keys()
    keys_ = np.array(list(keys_))    
    size_sessions = len(spiketimes_all_sessions)

    waveform_len = waveform_len * Fs    # in samples
    TRANSIENT_AMPLITUDE_VALID_DURATION = 7e-4 # seconds (duration of data before and after each spike that we consider when deciding the transient amplitude)
    TAVD_NSAMPLE = int(np.ceil(TRANSIENT_AMPLITUDE_VALID_DURATION*Fs))

    for iter_session in range(size_sessions):
        tmp_spk_stamp = spiketimes_all_sessions[keys_[iter_session]].astype(int)    # spike times
        
        tmp_spk_stamp = tmp_spk_stamp[(tmp_spk_stamp>=int((waveform_len-1)/2)) & (tmp_spk_stamp<=(filt_mda_single_channel.shape[0]-1-int(waveform_len/2)))]
        tmp_spk_start = tmp_spk_stamp - int((waveform_len-1)/2)
        waveforms_this_cluster = filt_mda_single_channel[np.array(tmp_spk_start[:,None]+np.arange(waveform_len),dtype = np.int64)] # (n_events, n_sample)
        waveforms_this_cluster_avg = np.mean(waveforms_this_cluster,axis = 0)
        waveforms_all.append(waveforms_this_cluster_avg)    # by session
        
        # Amplitude histogram computation here
        waveform_peaks   = np.max(waveforms_this_cluster[:, int(waveform_len//2-TAVD_NSAMPLE):int(waveform_len//2+TAVD_NSAMPLE)],axis = 1) 
        waveform_troughs = np.min(waveforms_this_cluster[:, int(waveform_len//2-TAVD_NSAMPLE):int(waveform_len//2+TAVD_NSAMPLE)],axis = 1)
        tmp_amp_series = (waveform_peaks-waveform_troughs) * (1-2*(waveform_peaks<0))
        spk_amp_series.append(tmp_amp_series)   # by session
        
    dict_out_amp_hist = {key:value for key,value in zip(keys_,spk_amp_series)}      # for a S.U ordered by session
    dict_out_all_waveforms = {key:value for key,value in zip(keys_,waveforms_all)}  # for a S.U ordered by session
    
    return (dict_out_amp_hist,dict_out_all_waveforms)

def extract_isi(input_dict,Fs,folder_save):    
    # This function is used to plot the ISI for a single unit 
    # input_dict: contains spike times in samples arranged by session
    # Fs: the sampling rate  
    keys_ = input_dict.keys()
    keys_ = np.array(list(keys_))    
    
    size_sessions = len(input_dict)
    n_bins=100
    bin_edges = np.linspace(0, 100, n_bins+1)    # bin edges for the ISI histogram
    isi_hist_all_sessions = []  # the ISI is 0 to 100 ms with a bin size of 1 ms
    
    for iter_l in range(size_sessions):
        isi_ = 1000*np.diff(input_dict[keys_[iter_l]])/Fs    #(in ms)
        isi_hist, edges = np.histogram(isi_,bin_edges)
        isi_hist_all_sessions.append(isi_hist)

        plt.figure()
        plt.plot(edges[:-1], isi_hist)
        plt.savefig(os.path.join(folder_save,f'_day_{keys_[iter_l]}_ISI.png'))
        plt.close()
    
    dict_output = {key:value for key,value in zip(keys_,isi_hist_all_sessions)}
    return dict_output
        

def extract_acg(input_dict,Fs,folder_save):    
    # This function is used to plot the ACG for a single unit 
    # input_dict: contains spike times in samples arranged by session
    # Fs: the sampling rate  
    keys_ = input_dict.keys()
    keys_ = np.array(list(keys_))    
    
    size_sessions = len(input_dict)
    acg_hist_all_sessions = []  # the ACG is of duration of 100 ms with a bin size of 0.5 ms
    
    duration = 0.1  # in seconds
    binSize = 0.0005    # in seconds
    half_bins = int(np.rint(duration/binSize/2))   # Since we are using these values in the matlab code (hard-coded)
    nBins = int(2*half_bins+1)
    time_axis = np.linspace(-half_bins,half_bins,nBins)*binSize
    
    eng = matlab.engine.start_matlab()
    eng.cd(r'../../Connectivity Analysis/', nargout=0)

    for iter_l in range(size_sessions):
        firings_local = input_dict[keys_[iter_l]]
        if firings_local.shape[0] == 0:
            acg_output_arr_np = np.zeros(time_axis.shape)
        else:
            firings_local = matlab.int64(firings_local)
            acg_output_arr = eng.wraper_calc_ACG_metrics(firings_local,Fs,nargout=1)
            acg_output_arr_np = acg_output_arr.tomemoryview().tolist()
            acg_output_arr_np = np.array(acg_output_arr_np,dtype = np.float)
            acg_output_arr_np = np.squeeze(acg_output_arr_np)

        acg_hist_all_sessions.append(acg_output_arr_np)

    eng.quit()
    dict_output = {key:value for key,value in zip(keys_,acg_hist_all_sessions)}
    return dict_output

def func_acg_extract_main(session_folder):

    def generate_hist_from_spiketimes(start_sample,end_sample, spike_times_local, window_in_samples):
        # n_windows_in_trial = int(np.ceil(end_sample-start_sample/window_in_samples))
        bin_edges = np.arange(start_sample, end_sample, step=window_in_samples)
        frq, edges = np.histogram(spike_times_local,bin_edges)
        return frq, edges

    def func_create_dict(sessions_label_stroke,spike_time_local,session_sample_abs):
        
        session_sample_abs_tmp = np.insert(session_sample_abs,0,0)
        lst_sessions_spike = []
        for iter_l in range(session_sample_abs.shape[0]):    # loop over sessions
            spike_time_singlesession = spike_time_local[np.logical_and(spike_time_local < session_sample_abs_tmp[iter_l+1],spike_time_local > session_sample_abs_tmp[iter_l] ) ]     # single session
            lst_sessions_spike.append(spike_time_singlesession)
        dict_local = dict(zip(sessions_label_stroke, lst_sessions_spike))
        
        
        return dict_local   # spike times by session



    global TRIAL_SESSION_MASK
    global TRIAL_KEEP_MASK
    global result_folder
    global result_folder_FR_avg
    global session_files_list
    # Input parameters ---------------------
    # session folder is a single measurement 
    parent_dir = os.path.abspath(os.path.join(session_folder, os.pardir))
    # firing rate calculation params
    WINDOW_LEN_IN_SEC = 2
    SMOOTHING_SIZE = 11
    DURATION_OF_INTEREST = 2.5  # how many seconds to look at upon stim onset (this is the activation or inhibition window)
    # Setting up
    session_trialtimes = os.path.join(session_folder,'trials_times.mat')
    trial_mask_file = os.path.join(session_folder,'trial_mask.csv')
    sessions_file = os.path.join(session_folder,'RHDfile_samples.csv')    # RHD samples for files (all sessions) [ending sample of each file if you ignore first entry]
    result_folder = os.path.join(session_folder,'Processed', 'count_analysis')
    result_folder_FR_avg = os.path.join(session_folder,'Processed', 'FR_clusters')
    result_folder_imp_clusters = os.path.join(session_folder,'Processed', 'important_clusters')
    sessions_label_stroke = os.path.join(parent_dir,'Sessions.csv')
    interesting_cluster_ids_file = os.path.join(session_folder,'interesting_clusters_.csv')  # Manually selected cluster IDs from PHY. These could be used as representative examples
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
    n_stim_end =int( stim_end_time * Fs )
    time_seq = float(data_pre_ms['SequenceTime'])
    Seq_perTrial = float(data_pre_ms['SeqPerTrial'])
    total_time = time_seq * Seq_perTrial
    print('Each sequence is: ', time_seq, 'sec')
    time_seq = int(np.ceil(time_seq * Fs/2))

    if os.path.isfile(sessions_file):
        session_files_list = pd.read_csv(sessions_file, header=None, index_col=False,dtype = np.uint32)
        session_files_list = session_files_list.to_numpy(dtype = np.uint32)
        session_files_list_session = np.squeeze(session_files_list[:,1].astype(np.int8))   # the session to which each trial belongs
        session_files_list_sample = np.squeeze(session_files_list[:,0])      # Trial accept mask    
        sessions_ids = np.unique(session_files_list_session)
        samples_start_session = []
        # for i_iter in sessions_ids:
        #     mask_session_local1 = (session_files_list_session == i_iter)
        #     mask_session_local = np.where(mask_session_local1)[0]
        #     np_fr_single_session = firing_rate_series[mask_session_local,:]
        #     np_frr_single_session = list(compress(firing_rate_rasters, mask_session_local1))
        
    else:
        Warning('WARNING: RHDfile_samples.csv not found!\n ')
    if os.path.isfile(sessions_label_stroke):
        sessions_label_stroke = pd.read_csv(sessions_label_stroke,header=None, index_col=False)
        sessions_label_stroke = sessions_label_stroke.iloc[:,0].to_list()
    else:
        Warning('WARNING: Sessions.csv not found!\n ')
    if os.path.isfile(interesting_cluster_ids_file):
        interesting_cluster_ids = pd.read_csv(interesting_cluster_ids_file,header = None , index_col= False)
        interesting_cluster_ids = interesting_cluster_ids.iloc[:,0].to_list()

    else:
        interesting_cluster_ids = np.array([0],dtype = np.int16)
        Warning('WARNING: interesting_clusters_.csv not found!\n ')

    trial_duration_in_samples = int(total_time*F_SAMPLE)
    window_in_samples = int(WINDOW_LEN_IN_SEC*F_SAMPLE)
    
    indx_rhdsessions_file = np.squeeze(np.where(np.diff(session_files_list_session)))
    session_sample_abs = np.append(session_files_list_sample[indx_rhdsessions_file],[session_files_list_sample[-1]])    # the ending sample of each session
    # Handle the exception
    firings = readmda(os.path.join(session_folder, "firings_clean_merged.mda")).astype(np.int64)
    # get spike stamp for all clusters (in SAMPLEs not seconds)
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
    n_clus = np.max(spike_labels)               # spike labels must start from 1
    print('Total number of clusters found: ',n_clus)
    spike_times_by_clus =[[] for i in range(n_clus)]
    spike_count_by_clus = np.zeros((n_clus,))
    for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
        spike_times_by_clus[spk_lbl-1].append(spk_time)
    for i in range(n_clus):
        spike_times_by_clus[i] = np.array(spike_times_by_clus[i])
        spike_count_by_clus[i] = spike_times_by_clus[i].shape[0]
    
    if os.path.isfile(os.path.join(session_folder, "filt.mda")):
        filt_signal = readmda(os.path.join(session_folder, "filt.mda")) # caution! big file
        filt_signal = filt_signal - np.mean(filt_signal,axis = 0)   # CMR (common mode rejected)
    else:
        Warning('WARNING: File filt.mda is missing! \n ------------')

    lst_waveforms_all = []
    lst_amplitudes_all = []
    lst_cluster_depth = []
    lst_isi_all = []

    # Read cluster locations
    clus_loc = pd.read_csv(os.path.join(session_folder,'clus_locations_clean_merged.csv'),header = None)
    clus_loc = clus_loc.to_numpy()

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

    cluster_all_range = np.arange(0,n_clus)
    interesting_cluster_ids = cluster_all_range[interesting_cluster_ids]
    lst_acg_all = []
    lst_filtered_data = []
    lst_FR_avg = []     # for representative only
    lst_x_ticks = []    # for representative only
    lst_FR_avg_all = [] # for all units
    lst_x_ticks_all = [] # for all units    
    
    for i_clus in range(n_clus):
        
        # Extracting plots for all units (repeated poorly written code due to time crunch)
        spike_time_local = spike_times_by_clus[i_clus]
        dict_local_i_clus = func_create_dict(sessions_label_stroke,spike_time_local,session_sample_abs)     # FR of a single unit by session
    
        session_sample_abs_tmp = np.insert(session_sample_abs,0,0)
    
        # histogram binning of the FR
        thiscluster_hist = []
        thiscluster_edges = []
        for iter_l in range(len(dict_local_i_clus)):    # loop over sessions
            start_sample = session_sample_abs_tmp[iter_l]
            end_sample = session_sample_abs_tmp[iter_l+1]
            hist_local, hist_edges_local = generate_hist_from_spiketimes(start_sample,end_sample, dict_local_i_clus[sessions_label_stroke[iter_l]], window_in_samples)
            thiscluster_hist.append(hist_local)
            thiscluster_edges.append(hist_edges_local)
            # plt.bar(hist_edges_local[:-1], hist_local)
        FR_mean_session = [np.mean(local_hist) for local_hist in thiscluster_hist]    
        FR_mean_session = np.array(FR_mean_session,dtype = float) * Fs/window_in_samples
        FR_mean_session = FR_mean_session.tolist()
        x_ticks = list(dict_local_i_clus.keys())
        lst_FR_avg_all.append(FR_mean_session)
        lst_x_ticks_all.append(x_ticks)
        
        depth = int(clus_loc[i_clus,1])  # triangulation by Jiaao

        # Extracting plots for only important representative single units
        if np.isin(i_clus,interesting_cluster_ids) and os.path.isfile(interesting_cluster_ids_file):
            
            spike_time_local = spike_times_by_clus[i_clus]
            dict_local_i_clus = func_create_dict(sessions_label_stroke,spike_time_local,session_sample_abs)     # FR of a single unit by session
        
            session_sample_abs_tmp = np.insert(session_sample_abs,0,0)
        
            # histogram binning of the FR
            thiscluster_hist = []
            thiscluster_edges = []
            for iter_l in range(len(dict_local_i_clus)):    # loop over sessions
                start_sample = session_sample_abs_tmp[iter_l]
                end_sample = session_sample_abs_tmp[iter_l+1]
                hist_local, hist_edges_local = generate_hist_from_spiketimes(start_sample,end_sample, dict_local_i_clus[sessions_label_stroke[iter_l]], window_in_samples)
                thiscluster_hist.append(hist_local)
                thiscluster_edges.append(hist_edges_local)
                # plt.bar(hist_edges_local[:-1], hist_local)
            FR_mean_session = [np.mean(local_hist) for local_hist in thiscluster_hist]    
            FR_mean_session = np.array(FR_mean_session,dtype = float) * Fs/window_in_samples
            FR_mean_session = FR_mean_session.tolist()
            x_ticks = list(dict_local_i_clus.keys())
            lst_FR_avg.append(FR_mean_session)
            lst_x_ticks.append(x_ticks)
            
            
            # ACG for each session (this unit)
            local_folder_create = result_folder_imp_clusters
            output_dict_acg = extract_acg(dict_local_i_clus,Fs,local_folder_create)
            lst_acg_all.append(output_dict_acg)
        
        # Extracting plots for only important representative single units
        if np.isin(i_clus,interesting_cluster_ids) and os.path.isfile(interesting_cluster_ids_file):
            # ISI for each session (this unit)
            local_folder_create = os.path.join(result_folder_imp_clusters,f'clusterid_{i_clus}')
            if not os.path.isdir(local_folder_create):
                os.makedirs(local_folder_create)
            output_dict_isi = extract_isi(dict_local_i_clus,Fs,local_folder_create)
            lst_isi_all.append(output_dict_isi)
            # Waveform on shank (this unit)
            (output_dict_amp,output_dict_waveforms) = extract_waveforms_and_amplitude(filt_signal[pri_ch_lut[i_clus],:],dict_local_i_clus,Fs)
            lst_waveforms_all.append(output_dict_waveforms)
            lst_amplitudes_all.append(output_dict_amp)
            lst_cluster_depth.append(depth)
    
    f1 = os.path.join(result_folder_FR_avg,'FR_avg_by_session.npy')     # Saves all units FR avg
    np.save(f1,lst_FR_avg_all)      
    f2 = os.path.join(result_folder_FR_avg,'sessions_all.npy')
    np.save(f2,lst_x_ticks_all)
    np.save(os.path.join(result_folder_imp_clusters,'amplitude_hist.npy'),lst_amplitudes_all) # primarily used for representative examples
    np.save(os.path.join(result_folder_imp_clusters,'waveforms_all.npy'),lst_waveforms_all)     # primarily used for representative examples
    np.save(os.path.join(result_folder_imp_clusters,'clus_depth.npy'),lst_cluster_depth)    # primarily used for representative examples
    np.save(os.path.join(result_folder_imp_clusters,'ISI_hist_all.npy'),lst_isi_all)    # primarily used for representative examples
    
    
    if os.path.isfile(interesting_cluster_ids_file):    # primarily used for representative examples
        f1 = os.path.join(result_folder_imp_clusters,'FR_avg_by_session.npy')
        np.save(f1,lst_FR_avg)
        f2 = os.path.join(result_folder_imp_clusters,'sessions_all.npy')
        np.save(f2,lst_x_ticks)
        # np.save(os.path.join(result_folder_imp_clusters,'ACG_hist_all.npy'),lst_acg_all)   
        


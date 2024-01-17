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
    
    cluster_all_range = np.arange(0,n_clus)
    interesting_cluster_ids = cluster_all_range[interesting_cluster_ids]
    lst_acg_all = []
    lst_filtered_data = []
    lst_FR_avg = []
    lst_x_ticks = []
    for i_clus in range(n_clus):
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
            
    if os.path.isfile(interesting_cluster_ids_file):
        f1 = os.path.join(result_folder_imp_clusters,'FR_avg_by_session.npy')
        np.save(f1,lst_FR_avg)
        f2 = os.path.join(result_folder_imp_clusters,'sessions_all.npy')
        np.save(f2,lst_x_ticks)
        # np.save(os.path.join(result_folder_imp_clusters,'ACG_hist_all.npy'),lst_acg_all)    # primarily used for representative examples


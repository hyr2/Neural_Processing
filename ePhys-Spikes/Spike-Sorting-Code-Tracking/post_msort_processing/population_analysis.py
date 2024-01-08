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
from adaptation import adaptation
sys.path.append('/home/hyr2-office/Documents/git/Neural_SP/Unreleased-Code-ePhys/bursts/')
from burst_analysis import *
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)
import skfda
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)


# Plotting fonts
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('font', size=16)          # controls default text sizes

def raster_all_trials(firing_stamp, t_trial_start, trial_duration_in_samples, window_in_samples):
    # returns the spike bin counts; need to manuaaly divide result by window length to get real firing rate
    n_windows_in_trial = int(np.ceil(trial_duration_in_samples/window_in_samples))
    bin_edges = np.arange(0, window_in_samples*n_windows_in_trial+1, step=window_in_samples)
    n_trials = t_trial_start.shape[0]
    # print(n_trials,Ntrials)
    # firing_rate_series_by_trial = np.zeros((n_trials, n_windows_in_trial))
    raster_series = []
    for i in range(n_trials):
        trial_firing_mask = np.logical_and(firing_stamp>=t_trial_start[i], firing_stamp<=(t_trial_start[i]+trial_duration_in_samples))
        trial_firing_stamp = firing_stamp[trial_firing_mask] - t_trial_start[i]
        trial_firing_stamp = trial_firing_stamp/trial_duration_in_samples * 13.5 # time axis raster plot
        # if trial_firing_stamp.shape[0]==0:
        #     continue
        raster_series.append(trial_firing_stamp)
    return raster_series

# Useful function for enumerated product for loop (itertools)
def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))

def FR_classifier_classic_zscore(firing_rate_avg_zscore,t_axis,t_range,stim_range):
    y_values = firing_rate_avg_zscore[t_range[0]:t_range[1]]
    t_axis = t_axis[t_range[0]:t_range[1]]
    y_values = np.reshape(y_values,[len(y_values),1])
    # second normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y_values)
    normalized = scaler.transform(y_values)
    
    # template
    bsl_template = np.mean(normalized[0:stim_range[0]-t_range[0]])
    # template_act = bsl_template * np.ones(y_values.shape,dtype = float)
    # template_sup = bsl_template * np.ones(y_values.shape,dtype = float)
    # template_act[stim_range[0]-t_range[0]+1:stim_range[1]-t_range[0]] = 1 
    # template_sup[stim_range[0]-t_range[0]+1:stim_range[1]-t_range[0]] = 0
    
    
    # max_zscore_stim = np.amax(np.absolute(firing_rate_avg_zscore[stim_range[0]:stim_range[1]]))
    max_z = np.amax(firing_rate_avg_zscore[stim_range[0]:stim_range[1]])
    min_z = np.amin(firing_rate_avg_zscore[stim_range[0]:stim_range[1]])
    # # Using step function template
    # SsD = np.zeros([2,])
    # SsD[0] = np.linalg.norm((normalized-template_act),2)
    # SsD[1] = np.linalg.norm((normalized-template_sup),2)

    
    # Gaussian template (distance metric)
    x1 = 1.4
    x2 = 4.5
    mu = 3.1        # center of peak
    sigma = 0.6    # width of gaussian
    z1 = (x1 - mu)/sigma
    z2 = (x2 - mu)/sigma
    x = np.linspace(z1,z2, len(t_axis))
    y = norm.pdf(x,0,1)
    y = y/np.amax(y)
    y = np.reshape(y,[len(y),1])
    scaler = MinMaxScaler(feature_range = (bsl_template,1))
    scaler = scaler.fit(y)
    template_act = scaler.transform(y)
    y = -y + 1
    scaler = MinMaxScaler(feature_range = (0,bsl_template))
    scaler = scaler.fit(y)
    template_sup = scaler.transform(y)
    
    SsD = np.zeros([2,])
    SsD[0] = np.linalg.norm((normalized-template_act),2)
    SsD[1] = np.linalg.norm((normalized-template_sup),2)
    
    # for idx, pair in enumerated_product(np.linspace(2.3,3.3,5), np.linspace(0.3,0.8,5)):
    #     print(idx, pair)
    #     mu = pair[0]
    #     sigma = pair[1]
    #     # calculate the z-transform
    #     z1 = ( x1 - mu ) / sigma
    #     z2 = ( x2 - mu ) / sigma
    #     x = np.linspace(z1, z2, len(t_axis))
    #     y = norm.pdf(x,0,1)
    #     y = y/np.amax(y)
        
    #     loc_x = idx[0]
    #     loc_y = idx[1]
        
    #     # activated neurons
    #     SaD[0,loc_x,loc_y] = np.sum(np.abs(y - normalized))
    #     SsD[0,loc_x,loc_y] = np.linalg.norm((y-normalized),2)
        
    #     # flipped (suppressed neurons)
    #     y = -y + 1
        
    #     SaD[1,loc_x,loc_y] = np.sum(np.abs(y - normalized))
    #     SsD[1,loc_x,loc_y] = np.linalg.norm((y-normalized),2)
            
    
    if np.abs(SsD[0] - SsD[1]) < 0.4:   # ie the SsD are very close to each other
        if (np.abs(max_z) > np.abs(min_z)):
            indx = 0    # activated
        elif (np.abs(max_z) < np.abs(min_z)):
            indx = 1    # suppressed
    else:
        indx = SsD.argmin()
    
    # override (necassary for short burst)
    if (np.abs(max_z) > 2.25*np.abs(min_z)): 
        indx = 0    # activated
    # indx = np.unravel_index(SsD.argmin(),SsD.shape)
    
    
    # plt.plot(template_sup)
    # plt.plot(template_act)
    # plt.plot(normalized)    
    # indx = np.unravel_index(SsD.argmin(),SsD.shape)
    
    # mu = np.linspace(2.3,3.3,10)[indx[1]]
    # sigma = np.linspace(2.3,3.3,10)[indx[2]]
    
    # # calculate the z-transform
    # z1 = ( x1 - mu ) / sigma
    # z2 = ( x2 - mu ) / sigma
    # x = np.linspace(z1, z2, len(t_axis))
    # y = norm.pdf(x,0,1)
    # y = y/np.amax(y) 
    
    # plt.plot(t_axis,normalized)
    # plt.plot(t_axis,y)
    # plt.plot(t_axis,-y+1)
    
    # plt.figure()
    # plt.plot(t_axis,normalized)
    # if indx == 0:
    #     plt.plot(t_axis,template_act)
    # else:
    #     plt.plot(t_axis,template_sup)
    
    return (t_axis,y_values,normalized,indx)

def extract_waveforms():
    pass

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

        # plt.figure()
        # plt.plot(edges[:-1], isi_hist)
        # plt.savefig(os.path.join(folder_save,f'_day_{keys_[iter_l]}_ISI.png'))
        # plt.close()
    
    dict_output = {key:value for key,value in zip(keys_,isi_hist_all_sessions)}
    return dict_output
        
        # frq, edges = np.histogram(spike_times_local,bin_edges)
        

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


def func_pop_analysis(session_folder,CHANNEL_MAP_FPATH):
    
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
   
    # read trials times
    trials_start_times = loadmat(session_trialtimes)['t_trial_start'].squeeze()
    
    # Trial mask (only for newer animals)
    if os.path.isfile(trial_mask_file):
        trial_mask = pd.read_csv(trial_mask_file, header=None, index_col=False,dtype = np.int8)
        TRIAL_KEEP_MASK = trial_mask.to_numpy(dtype = np.int8)
        TRIAL_SESSION_MASK = np.squeeze(TRIAL_KEEP_MASK[:,1])   # the session to which each trial belongs
        TRIAL_KEEP_MASK = np.squeeze(TRIAL_KEEP_MASK[:,0]).astype('bool')      # Trial accept mask 
    else:
        TRIAL_KEEP_MASK = np.ones([trials_start_times.shape[0],],dtype = bool)
        Warning('WARNING: Trial mask not found!\n ')
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
        os.makedirs(result_folder_imp_clusters)
    else:
        Warning('WARNING: interesting_clusters_.csv not found!\n ')
    
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
    try:
        if n_ch_this_session == Num_chan:
            pass
        else:
            raise ValueError("Number of channels mismatch b/w geom.csv and pre_MS.json!\n")
    except ValueError as e: 
        print(e)
        
    indx_rhdsessions_file = np.squeeze(np.where(np.diff(session_files_list_session)))
    session_sample_abs = np.append(session_files_list_sample[indx_rhdsessions_file],[session_files_list_sample[-1]])    # the ending sample of each session
    # Handle the exception
    firings = readmda(os.path.join(session_folder, "firings_clean_merged.mda")).astype(np.int64)
    # get spike stamp for all clusters (in SAMPLEs not seconds)
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
    spike_labels_unique = np.unique(spike_labels)
    single_unit_mask = np.ones(spike_labels_unique.shape,dtype = bool)  # redundant
    n_clus = np.max(spike_labels)               # spike labels must start from 1
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
    burst_cfgs = {}


    def get_single_cluster_spikebincouts_all_trials(firing_stamp, t_trial_start, trial_duration_in_samples, window_in_samples):
        global TRIAL_KEEP_MASK
        global TRIAL_SESSION_MASK
        # returns the spike bin counts; need to manuaaly divide result by window length to get real firing rate
        n_windows_in_trial = int(np.ceil(trial_duration_in_samples/window_in_samples))
        bin_edges = np.arange(0, window_in_samples*n_windows_in_trial+1, step=window_in_samples)
        n_trials = t_trial_start.shape[0]
        n_sessions = np.unique(TRIAL_SESSION_MASK).shape[0]
        # print(n_trials,Ntrials)
        assert n_trials==Ntrials or n_trials==Ntrials+1, "%d %d" % (n_trials, Ntrials)
        if n_trials > Ntrials:
            n_trials = Ntrials
        firing_rate_series_by_trial = np.zeros((n_trials, n_windows_in_trial))
        for i in range(n_trials):
            local_session_ID = TRIAL_SESSION_MASK[i]
            trial_firing_mask = np.logical_and(firing_stamp>=t_trial_start[i], firing_stamp<=(t_trial_start[i]+trial_duration_in_samples))
            trial_firing_stamp = firing_stamp[trial_firing_mask] - t_trial_start[i]
            if trial_firing_stamp.shape[0]==0:
                continue
            tmp_hist, _ = np.histogram(trial_firing_stamp, bin_edges)   # The firing rate series for each trial for a single cluster
            firing_rate_series_by_trial[i,:] = tmp_hist   
        return firing_rate_series_by_trial

    def single_cluster_main(i_clus):
        global TRIAL_KEEP_MASK
        global TRIAL_SESSION_MASK
        global result_folder_FR_avg
        
        firing_stamp = spike_times_by_clus[i_clus] 
        N_spikes_local = spike_times_by_clus[i_clus].size
        
        stim_start_time =  n_stim_start     # number of samples after which stim begins
        stim_end_time = n_stim_start + (Fs * 1.5) # 1.5 s post stim start is set as the stim end time. We only consider the first 1500ms 
        bsl_start_time = Fs*0.5     # start of baseline period
        # Spike time stamps 
        # burst_cfgs["detection_method"] = "log_isi"
        # burst_cfgs["min_spikes_per_burst"] = 3
        # burst_cfgs["max_short_isi_ms"] = 0.050  # 50 ms (after empirical observations)
        # burst_cfgs["max_long_isi_ms"] = 0.120   # 120ms (after empirical observations)
        
        #burst_dict = SingleUnit_burst(firing_stamp,trials_start_times,stim_start_time,stim_end_time,bsl_start_time,Fs,TRIAL_KEEP_MASK,burst_cfgs)   # **** needs updates to multiple sessions
        window_in_time = 0.1
        window_in_samples_n = window_in_time * Fs
        # Firing rate series (for each session ****)
        firing_rate_series = get_single_cluster_spikebincouts_all_trials(
            firing_stamp, 
            trials_start_times, 
            trial_duration_in_samples, 
            window_in_samples_n
            )
        
        t_axis = np.linspace(0,firing_rate_series.shape[1]*window_in_time,firing_rate_series.shape[1])
        t_2 = np.squeeze(np.where(t_axis >= 4.025))[0]      
        t_1 = np.squeeze(np.where(t_axis >= 2.525))[0]      
        
        firing_rate_rasters = raster_all_trials(firing_stamp, trials_start_times, trial_duration_in_samples, window_in_samples)
        
        # Splitting into sessions
        fr_series_ids = []  # size should be equal to the number of sessions
        fr_rasters_ids = []
        trial_keep_session = []
        sessions_ids = np.unique(TRIAL_SESSION_MASK)
        for i_iter in sessions_ids:
            mask_session_local1 = (TRIAL_SESSION_MASK == i_iter)
            mask_session_local = np.where(mask_session_local1)[0]
            np_fr_single_session = firing_rate_series[mask_session_local,:]
            np_frr_single_session = list(compress(firing_rate_rasters, mask_session_local1))
            
            trial_keep_session.append(TRIAL_KEEP_MASK[mask_session_local])
            fr_series_ids.append(np_fr_single_session)
            fr_rasters_ids.append(np_frr_single_session)
        
        # firing rate series for a single session (trial mask applied)
        firing_rate_series = []
        firing_rate_avg_allsessions = []
        np_arr_session_binary = np.zeros([sessions_ids.shape[0],],dtype = bool)     # Did this session spike (output boolean array)
        np_arr_session_binary_up_down = np.zeros([sessions_ids.shape[0],],dtype = np.int8)  # was the FR higher or lower chronically w.r.t baseline
        dict_spikes = {
            'FR_avg_spont' : [],                                                                                                # trial mask not applied
            'S_total_spont' : [],     # Here I need to add MUA as well                                                          # trial mask not applied
            'FR_avg_stim' : [],                                                                                                 # trial mask is applied
            'S_total_stim' : []       # Here I need to add MUA as well (this should be total spikes during whisker deflection)  # trial mask is applied
            }
        for iter_l in sessions_ids:
            fr_local = fr_series_ids[iter_l]
            
            dict_spikes['FR_avg_spont'].append(np.mean(fr_local,axis = (0,1))/window_in_time)               # Avg FR per trial of this unit
            dict_spikes['S_total_spont'].append(np.mean(np.sum(fr_local,axis = 1)))                         # Total spikes per trial of this unit
            
            
            trial_mask_local = trial_keep_session[iter_l].astype('bool')
            fr_local = fr_local[trial_mask_local,:]
            firing_rate_series.append(fr_local/window_in_time)
            fr_rasters_local = fr_rasters_ids[iter_l]
            
            dict_spikes['FR_avg_stim'].append(np.mean(fr_local[:,t_1:t_2],axis = (0,1))/window_in_time)     # Avg FR per trial of this unit
            dict_spikes['S_total_stim'].append(np.mean(np.sum(fr_local[:,t_1:t_2],axis = 1)))                # Total spikes per trial of this unit
            
            # plotting rasters by session
            # plt.figure()
            # for iter_t in range(len(fr_rasters_local)):
            #     raster_local = fr_rasters_local[iter_t]
            #     y_local = iter_t + np.ones(raster_local.shape)
            #     plt.plot(raster_local,y_local,color = 'blue',marker = "o",linestyle = 'None',markersize = 3)
            # plt.xlim(2,3.5)
            # plt.axis('off')
            # filename_save = os.path.join(result_folder_FR_avg,f'raster__cluster{i_clus+1}_session{iter_l}.png')
            # plt.savefig(filename_save,format = 'png')
            # plt.close()
            
            # plotting avg FR by session
            firing_rate_avg = np.mean(fr_local, axis=0) # averaging over all trials
            firing_rate_avg_nofilt = firing_rate_avg/window_in_time                     # in Hz
            firing_rate_avg = filter_Savitzky_slow(firing_rate_avg/window_in_time)      # in Hz
            # t_axis = np.linspace(0,firing_rate_avg.shape[0]*WINDOW_LEN_IN_SEC,firing_rate_avg.shape[0])
            # plt.figure()
            # plt.plot(t_axis,firing_rate_avg,linewidth  = 2.5,color = 'k',linestyle = '-')
            # plt.xlim([0,4])
            # plt.ylim([0,1])
            # filename_save = os.path.join(result_folder_FR_avg,f'FR__cluster{i_clus+1}_session{iter_l}.png')
            # plt.savefig(filename_save,format = 'png',dpi=250)
            # plt.close()
            
            
            firing_rate_avg_allsessions.append(firing_rate_avg)
            
            firing_rate_avg = np.mean(firing_rate_avg)
            if firing_rate_avg > 0.2: 
                np_arr_session_binary[iter_l] = True
            print(f'Avg FR for session {iter_l}: {firing_rate_avg}')
        
        #testing for significance in increased or decreased FR b/w sessions (first two are baselines)   [both tonic and phasic/ no regards for stimulation]
        bsl_FR = np.concatenate((firing_rate_avg_allsessions[0],firing_rate_avg_allsessions[1])) 
        chr_FR = np.concatenate(firing_rate_avg_allsessions[3:])
        
        if np.mean(bsl_FR) != 0 :
            Z_score_val = (np.mean(chr_FR) - np.mean(bsl_FR))/np.mean(bsl_FR) 
        else:
            Z_score_val = 1
        
        filename_save = os.path.join(os.path.join(result_folder_FR_avg,f'ChangeFR_{i_clus+1}.png'))
        f, axes = plt.subplots(1,2)
        axes=axes.flatten()
        f.tight_layout(pad=1.0) 
        sns.set_style('white')
        axes[0].hist(bsl_FR, bins=35, color='k', alpha=1)
        axes[0].hist(chr_FR, bins=35, color='k', alpha=0.25)
        sns.despine(f)
        hist, bins = np.histogram(chr_FR, bins=35, density=True)
        hist = hist/hist.sum()
        axes[1].plot(bins[:-1], hist, color='k',alpha = 0.3, linewidth = 3)
        hist, bins = np.histogram(bsl_FR, bins=35, density=True)
        hist = hist/hist.sum()
        axes[1].plot(bins[:-1], hist, color='k',alpha = 1,linewidth = 3)
        sns.despine(f)
        # f.suptitle(f'Z-Value: {Z_score_val}', fontsize=18)
        plt.savefig(filename_save,format = 'png')
        
        # Z_score_val = (np.mean(chr_FR) - np.mean(bsl_FR))/np.mean(bsl_FR) 
        if Z_score_val > 1.95:
            np_arr_session_binary_up_down = 1
        elif (Z_score_val < -0.25):
            np_arr_session_binary_up_down = -1
        else:
            #not significant
            np_arr_session_binary_up_down = 0 
                
        
        return (np_arr_session_binary_up_down, np_arr_session_binary,fr_series_ids,trial_keep_session,dict_spikes)
            
        # **************** check with raster. Something seems wrong in binning into histogram

        # firing_rate_series = firing_rate_series[TRIAL_KEEP_MASK, :]/WINDOW_LEN_IN_SEC       # This gives firing rate in Hz
        # firing_rate_avg = np.mean(firing_rate_series, axis=0) # averaging over all trials
        # firing_rate_avg_nofilt = firing_rate_avg
        # firing_rate_avg = filterSignal_lowpass(firing_rate_avg, np.single(1/WINDOW_LEN_IN_SEC), axis_value = 0)
        # firing_rate_avg = filter_Savitzky_slow(firing_rate_avg)
        # firing_rate_sum = np.sum(firing_rate_series, axis=0)
        
        # n_samples_baseline = int(np.ceil(stim_start_time/WINDOW_LEN_IN_SEC))
        # n_samples_stim = int(np.ceil((stim_end_time-stim_start_time)/WINDOW_LEN_IN_SEC))
        
        # t_axis = np.linspace(0,firing_rate_avg.shape[0]*WINDOW_LEN_IN_SEC,firing_rate_avg.shape[0])
        # t_1 = np.squeeze(np.where(t_axis >= 1.4))[0]      # stim start set to 2.45 seconds
        # t_2 = np.squeeze(np.where(t_axis >= 2.0))[0]      # end of bsl region (actual value is 2.5s but FR starts to increase before. Maybe anticipatory spiking due to training)
        # t_3 = np.squeeze(np.where(t_axis <= 2.1))[-1]        # stim end set to 5.15 seconds
        # t_4 = np.squeeze(np.where(t_axis <= 4.02))[-1]
        # t_5 = np.squeeze(np.where(t_axis <= 8))[-1]        # post stim quiet period
        # t_6 = np.squeeze(np.where(t_axis <= 9.5))[-1]
        # t_7 = np.squeeze(np.where(t_axis <= 4.5))[-1]
        # t_8 = np.squeeze(np.where(t_axis <= 2.520))[-1]       
        # t_9 = np.squeeze(np.where(t_axis <= 0.6))[-1]       
        # # Zscore
        # bsl_mean = np.mean(np.hstack((firing_rate_avg[t_1:t_2],firing_rate_avg[t_5:t_6]))) # baseline is pre and post stim
        # bsl_std = np.std(np.hstack((firing_rate_avg[t_1:t_2],firing_rate_avg[t_5:t_6])))   # baseline is pre and post stim 
        # firing_rate_zscore = zscore_bsl(firing_rate_avg, bsl_mean, bsl_std)          # Zscore to classify cluster as activated or suppressed
        
        # # FR Classify
        # (t_axis_s,firing_rate_zscore_s,normalized,indx) = FR_classifier_classic_zscore(firing_rate_zscore,t_axis,[t_1,t_7],[t_8,t_7])
        
        # max_zscore_stim = np.amax(np.absolute(firing_rate_zscore[t_8:t_7]))          # Thresholding for Z score values
        # # plt.figure()
        # plt.plot(t_axis,firing_rate_zscore)
        # firing_rate_zscore = zscore(firing_rate_avg, axis = 0)          # Zscore to classify cluster as activated or suppressed        
        # t_stat, pval_2t = ttest_ind(
        #     firing_rate_avg[t_1:t_2], 
        #     firing_rate_avg[t_3:t_4], 
        #     equal_var=False)
        # if t_stat > 0:
        #     # suppressed
        #     t_stat, pval_2t = ttest_ind(
        #         np.hstack((firing_rate_avg[t_1:t_2],firing_rate_avg[t_5:t_6])), 
        #         firing_rate_avg[t_3:t_4], 
        #         equal_var=False,alternative = 'greater')
        #     if pval_2t < 0.01 and N_spikes_local>(Ntrials*3) and max_zscore_stim > 2:   # sparse firing neuron is rejected + peak response needs to be 2 sd higher than bsl
        #         # reject null
        #         clus_property = ANALYSIS_INHIBITORY
        #     else:
        #         clus_property = ANALYSIS_NOCHANGE
        # else:
        #     # activated
        #     t_stat, pval_2t = ttest_ind(
        #         np.hstack((firing_rate_avg[t_1:t_2],firing_rate_avg[t_5:t_6])), 
        #         firing_rate_avg[t_3:t_4], 
        #         equal_var=False,alternative = 'less')                
        #     if pval_2t < 0.01 and N_spikes_local>(Ntrials*3) and max_zscore_stim > 3:   # sparse firing neuron is rejected + peak response needs to be 3 sd higher than bsl
        #         # reject null
        #         clus_property = ANALYSIS_EXCITATORY
        #     else:
        #         clus_property = ANALYSIS_NOCHANGE
        
        # Add t-test + add spike count difference as two new metrics for "activated" and "suppressed" neurons
        
        # max_z = np.amax((firing_rate_zscore[t_8:t_7]))
        # min_z = np.amin((firing_rate_zscore[t_8:t_7]))
        # clus_property_1 = 0
        # if max_zscore_stim > 2.5:
        #     if indx == 0 and N_spikes_local>(Ntrials*3) and (np.abs(max_z) > np.abs(min_z)):
        #         # print('activated neuron')
        #         clus_property_1 = 1
        #     elif indx == 1 and N_spikes_local>(Ntrials*3) and (np.abs(max_z) < np.abs(min_z)):
        #         # print('suppressed neuron')
        #         clus_property_1 = -1
        # # print(clus_property)
        # # if pval_2t > .01:
        # #     clus_property = ANALYSIS_NOCHANGE
        # # elif t_stat < 0:
        # #     clus_property = ANALYSIS_EXCITATORY
        # # else:
        # #     clus_property = ANALYSIS_INHIBITORY
        # # print(clus_property)
        
        # # Computing number of spikes
        # firing_rate_series2 = firing_rate_series * WINDOW_LEN_IN_SEC    # Number of spikes (histogram over for all trials)
        # firing_rate_series_avg = np.mean(firing_rate_series2,axis = 0)  # Avg number of spikes over accepted trials
        # Spikes_stim = np.sum(firing_rate_series_avg[t_8:t_4])   # 1.5 sec
        # Spikes_bsl = np.sum(firing_rate_series_avg[t_9:t_3])    # 1.5 sec
        # Spikes_num = np.array([Spikes_bsl,Spikes_stim])

        # return clus_property_1, firing_rate_avg, firing_rate_sum, Spikes_num, firing_rate_series, burst_dict


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
    
    
    def func_create_dict(sessions_label_stroke,spike_time_local,session_sample_abs):
        
        session_sample_abs_tmp = np.insert(session_sample_abs,0,0)
        lst_sessions_spike = []
        for iter_l in range(session_sample_abs.shape[0]):    # loop over sessions
            spike_time_singlesession = spike_time_local[np.logical_and(spike_time_local < session_sample_abs_tmp[iter_l+1],spike_time_local > session_sample_abs_tmp[iter_l] ) ]     # single session
            lst_sessions_spike.append(spike_time_singlesession)
        dict_local = dict(zip(sessions_label_stroke, lst_sessions_spike))
        
        
        return dict_local   # spike times by session
        
    def generate_hist_from_spiketimes(start_sample,end_sample, spike_times_local, window_in_samples):
        # n_windows_in_trial = int(np.ceil(end_sample-start_sample/window_in_samples))
        bin_edges = np.arange(start_sample, end_sample, step=window_in_samples)
        frq, edges = np.histogram(spike_times_local,bin_edges)
        return frq, edges
    
    lst_filtered_data = []
    
    if os.path.isfile(os.path.join(session_folder, "filt.mda")):
        filt_signal = readmda(os.path.join(session_folder, "filt.mda")) # caution! big file
        filt_signal = filt_signal - np.mean(filt_signal,axis = 0)   # CMR (common mode rejected)
    else:
        Warning('WARNING: File filt.mda is missing! \n ------------')

    lst_waveforms_all = []
    lst_amplitudes_all = []
    lst_cluster_depth = []
    lst_isi_all = []
    cluster_all_range = np.arange(0,n_clus)
    interesting_cluster_ids = cluster_all_range[interesting_cluster_ids]
    for i_clus in range(n_clus):
        
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
            # ax.bar(hist_edges_local[:-1], hist_local)
        # Filtering the signal
        all_FR_thiscluster = np.concatenate(thiscluster_hist, axis = 0)
        all_edges_thiscluster = np.concatenate(thiscluster_edges, axis = 0)
        x_ticks = [local_arr.shape[0] for local_arr in thiscluster_edges]
        x_ticks = [sum(x_ticks[:i+1]) for i in range(len(x_ticks))]
        
        Num_to_drop = all_edges_thiscluster.shape[0] - all_FR_thiscluster.shape[0]
        all_edges_thiscluster = all_edges_thiscluster[:-Num_to_drop]
        
        all_FR_thiscluster_f = np.abs(savgol_filter(all_FR_thiscluster,200,3,mode = 'nearest'))
        # fig, ax = plt.subplots()
        # ax.plot(all_FR_thiscluster_f)
        # ax.set_xticks(x_ticks)
        # ax.set_xticklabels(dict_local_i_clus.keys())
        
        lst_filtered_data.append(all_FR_thiscluster_f)
        
        # Need to add facilitating cell vs adapting cell vs no change cell
        # clus_property, firing_rate_avg, firing_rate_sum, Spikes_num, firing_rate_series, burst_dict = single_cluster_main(i_clus)
        ( plasticity_metric, arr_sessions_spiking, fr_series_ids_out,trial_keep_session_out,dict_all_sessions) = single_cluster_main(i_clus)
        
        # creating a dictionary for this cluster
        N_spikes_local = spike_time_local.size 
        # shank_num = int(clus_loc[i_clus,0] // 250)      # starts from 0
        depth = int(clus_loc[i_clus,1])  # triangulation by Jiaao
        # Get shank ID from primary channel for the cluster
        # shank_num = get_shanknum_from_msort_id(prim_ch)
        i_clus_dict  = {}
        i_clus_dict['cluster_id'] = i_clus + 1 # to align with the discard_noise_viz.py code cluster order [folder: figs_allclus_waveforms]
        i_clus_dict['total_spike_count'] = N_spikes_local
        i_clus_dict['depth'] =  depth
        i_clus_dict['plasticity_metric'] =  plasticity_metric
        i_clus_dict['sessions_visibility'] =  arr_sessions_spiking
        
        z_clus_dict = {**i_clus_dict, **dict_all_sessions}  # combining two dictionaries 
        
        # Append to output dataframe of all units in current session
        list_all_clus.append(z_clus_dict)
        
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
       
        iter_local = iter_local+1
        
    # 
    # For clustering and ML    
    # Scaling the data ()
    X_in = np.transpose(np.stack(lst_filtered_data))
    scaled_data_list = [
        StandardScaler().fit_transform(X_in),
        MaxAbsScaler().fit_transform(X_in),
        RobustScaler(quantile_range=(25, 75)).fit_transform(X_in)   # bad since outliers units are left intact ie their FR is much different to the rest
        ]
    # Plotting scaled outputs
    
    # randomly sample 100 datapoints from each phase of stroke
    x_ticks[-1] = X_in.shape[0]
    days_this_dataset = list(dict_local_i_clus.keys())
    days_this_dataset = np.array(days_this_dataset,dtype = np.int8)
    
    samples_temporal_phase = [x_ticks[np.where(days_this_dataset < 0)[0][-1]], x_ticks[np.where(days_this_dataset < 21)[0][-1]] , x_ticks[np.where(days_this_dataset < 28)[0][-1]] , x_ticks[-1] ]
     
    scaled_data_list_new = []
    for iter_l in range(len(scaled_data_list)):
        
        arr_local_tmp = scaled_data_list[iter_l][0:samples_temporal_phase[0],:]
        phase_bsl = np.array([np.random.choice(arr_local_tmp[:,iter_ll],200,replace=False) for iter_ll in range(arr_local_tmp.shape[1])],dtype = np.single)
        
        arr_local_tmp = scaled_data_list[iter_l][samples_temporal_phase[0]:samples_temporal_phase[1],:]
        phase_rec1 = np.array([np.random.choice(arr_local_tmp[:,iter_ll],100,replace=False) for iter_ll in range(arr_local_tmp.shape[1])],dtype = np.single)
        
        arr_local_tmp = scaled_data_list[iter_l][samples_temporal_phase[1]:samples_temporal_phase[2],:]
        phase_rec2 = np.array([np.random.choice(arr_local_tmp[:,iter_ll],100,replace=False) for iter_ll in range(arr_local_tmp.shape[1])],dtype = np.single)

        arr_local_tmp = scaled_data_list[iter_l][samples_temporal_phase[2]:samples_temporal_phase[3],:]
        phase_chr = np.array([np.random.choice(arr_local_tmp[:,iter_ll],400,replace=False) for iter_ll in range(arr_local_tmp.shape[1])],dtype = np.single)
        
        # combing all phases: 100 samples from pre-stroke, 100 samples from recovery phase and 100 samples from chronic phase post stroke
        phase_all_local = np.concatenate((phase_bsl,phase_rec1,phase_rec2,phase_chr),axis = 1)
        
        # filtering (lowpass)
        phase_all_local = savgol_filter(phase_all_local,17,3,mode = 'nearest',axis = 1)
        scaled_data_list_new.append(phase_all_local)
        # Creating FDataGrid object for sklearn-fda
        # scaled_data_list_new.append(skfda.FDataGrid(
        #     data_matrix = phase_all_local,
        #     grid_points = None
        #     ))
        
        
    
    # plt.figure()
    # # clus_response_mask  = np.squeeze(clus_response_mask.to_numpy())
    # clus_response_mask = np.squeeze(clus_response_mask)
    # for i_clus in range(n_clus):
    #     if single_unit_mask[i_clus] == True and clus_response_mask[i_clus] == 1 :
    #         firing_stamp = spike_times_by_clus[i_clus]/Fs
            
    #         firing_raster = firing_stamp[np.logical_and(firing_stamp>173.36,firing_stamp<186.86)]
    #         y1 = (i_clus+1)*np.ones(firing_raster.shape)
    #         plt.plot(firing_raster,y1,color = 'black',marker = ".",linestyle = 'None')
        

    # savemat(
    #     os.path.join(session_folder, "clusters_response_mask.mat"),
    #     {"clus_response_mask": clus_response_mask}
    # )

                
    
    # A mask for SUA and MUA
    # FR_series_all_clusters = np.squeeze(np.array(FR_series_all_clusters))
    # accept_mask_local = np.logical_or(single_unit_mask == True, multi_unit_mask == True)
    # FR_series_all_clusters = FR_series_all_clusters[single_unit_mask]  # Only save the accepted clusters
    
    # # Firing rate computation
    # Time_vec = np.linspace(0,13.5,FR_series_all_clusters[1].shape[0])
    # stim_start_idx = int(stim_start_time/WINDOW_LEN_IN_SEC)
    # doi_end_idx = int((stim_start_time+DURATION_OF_INTEREST)/WINDOW_LEN_IN_SEC)
    # stim_end_idx = int((stim_start_time+float(data_pre_ms['StimulationTime']))/WINDOW_LEN_IN_SEC)
    # # activated neurons
    # shankA_act = np.squeeze(FR_list_byshank_act[0])
    # shankA_act = np.reshape(shankA_act,[int(shankA_act.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    # shankB_act = np.squeeze(FR_list_byshank_act[1])
    # shankB_act = np.reshape(shankB_act,[int(shankB_act.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    # shankC_act = np.squeeze(FR_list_byshank_act[2])
    # shankC_act = np.reshape(shankC_act,[int(shankC_act.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    # shankD_act = np.squeeze(FR_list_byshank_act[3])
    # shankD_act = np.reshape(shankD_act,[int(shankD_act.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    # compute bsl FR
    # shankA_act_bsl = np.mean(shankA_act[:,:stim_start_idx],axis = 1) if np.size(shankA_act) != 0 else np.nan
    # shankB_act_bsl = np.mean(shankB_act[:,:stim_start_idx],axis = 1) if np.size(shankB_act) != 0 else np.nan
    # shankC_act_bsl = np.mean(shankC_act[:,:stim_start_idx],axis = 1) if np.size(shankC_act) != 0 else np.nan
    # shankD_act_bsl = np.mean(shankD_act[:,:stim_start_idx],axis = 1) if np.size(shankD_act) != 0 else np.nan
    # bsl normalized FR
    # shankA_act = (shankA_act / shankA_act_bsl[:, np.newaxis] - 1) if np.size(shankA_act) != 0 else np.nan
    # shankB_act = (shankB_act / shankB_act_bsl[:, np.newaxis] - 1) if np.size(shankB_act) != 0 else np.nan
    # shankC_act = (shankC_act / shankC_act_bsl[:, np.newaxis] - 1) if np.size(shankC_act) != 0 else np.nan
    # shankD_act = (shankD_act / shankD_act_bsl[:, np.newaxis] - 1) if np.size(shankD_act) != 0 else np.nan
    
    # shankA_act = zscore(shankA_act, axis = 1) if np.size(shankA_act) != 0 else np.nan
    # shankB_act = zscore(shankB_act, axis = 1) if np.size(shankB_act) != 0 else np.nan
    # shankC_act = zscore(shankC_act, axis = 1) if np.size(shankC_act) != 0 else np.nan
    # shankD_act = zscore(shankD_act, axis = 1) if np.size(shankD_act) != 0 else np.nan
    # average FR during activation
    # shankA_act = np.mean(np.mean(shankA_act[:,stim_start_idx:doi_end_idx],axis=1)) if not np.isnan(shankA_act).any() else np.nan  # average FR during activation
    # shankB_act = np.mean(np.mean(shankB_act[:,stim_start_idx:doi_end_idx],axis=1)) if not np.isnan(shankB_act).any()  else np.nan  # average FR during activation
    # shankC_act = np.mean(np.mean(shankC_act[:,stim_start_idx:doi_end_idx],axis=1)) if not np.isnan(shankC_act).any()  else np.nan  # average FR during activation
    # shankD_act = np.mean(np.mean(shankD_act[:,stim_start_idx:doi_end_idx],axis=1)) if not np.isnan(shankD_act).any()  else np.nan  # average FR during activation
    
    # supressed neurons
    # # supressed neurons
    # shankA_inh = np.squeeze(FR_list_byshank_inh[0])
    # shankA_inh = np.reshape(shankA_inh,[int(shankA_inh.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    # shankB_inh = np.squeeze(FR_list_byshank_inh[1])
    # shankB_inh = np.reshape(shankB_inh,[int(shankB_inh.size/Time_vec.shape[0]),Time_vec.shape[0]])
    # shankC_inh= np.squeeze(FR_list_byshank_inh[2])
    # shankC_inh = np.reshape(shankC_inh,[int(shankC_inh.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    # shankD_inh= np.squeeze(FR_list_byshank_inh[3])
    # shankD_inh = np.reshape(shankD_inh,[int(shankD_inh.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    # # compute bsl FR
    # # shankA_inh_bsl = np.mean(shankA_inh[:,:stim_start_idx],axis = 1) if np.size(shankA_inh) != 0 else np.nan
    # shankB_inh_bsl = np.mean(shankB_inh[:,:stim_start_idx],axis = 1) if np.size(shankB_inh) != 0 else np.nan
    # shankC_inh_bsl = np.mean(shankC_inh[:,:stim_start_idx],axis = 1) if np.size(shankC_inh) != 0 else np.nan
    # shankD_inh_bsl = np.mean(shankD_inh[:,:stim_start_idx],axis = 1) if np.size(shankD_inh) != 0 else np.nan
    # bsl normalized FR
    # shankA_inh = (shankA_inh/ shankA_inh_bsl[:, np.newaxis] - 1) if (np.size(shankA_inh) != 0) else np.nan
    # shankB_inh= (shankB_inh/ shankB_inh_bsl[:, np.newaxis] - 1) if np.size(shankB_inh) != 0 else np.nan
    # shankC_inh= (shankC_inh/ shankC_inh_bsl[:, np.newaxis] - 1) if np.size(shankC_inh) != 0 else np.nan
    # shankD_inh= (shankD_inh/ shankD_inh_bsl[:, np.newaxis] - 1) if np.size(shankD_inh) != 0 else np.nan
    
    # shankA_inh = zscore(shankA_inh, axis = 1) if (np.size(shankA_inh) != 0) else np.nan
    # shankB_inh = zscore(shankB_inh, axis = 1) if np.size(shankB_inh) != 0 else np.nan
    # shankC_inh = zscore(shankC_inh, axis = 1) if np.size(shankC_inh) != 0 else np.nan
    # shankD_inh = zscore(shankD_inh, axis = 1) if np.size(shankD_inh) != 0 else np.nan
    # average FR during suppression
    # shankA_inh= np.mean(np.amin(shankA_inh[:,stim_start_idx:stim_end_idx], axis = 1)) if not np.isnan(shankA_inh).any() else np.nan  # average FR during activation
    # shankB_inh= np.mean(np.amin(shankB_inh[:,stim_start_idx:stim_end_idx], axis = 1)) if not np.isnan(shankB_inh).any()  else np.nan  # average FR during activation
    # shankC_inh= np.mean(np.amin(shankC_inh[:,stim_start_idx:stim_end_idx], axis = 1)) if not np.isnan(shankC_inh).any()  else np.nan  # average FR during activation
    # shankD_inh= np.mean(np.amin(shankD_inh[:,stim_start_idx:stim_end_idx], axis = 1)) if not np.isnan(shankD_inh).any()  else np.nan  # average FR during activation

    # Output .mat file
    # pd.DataFrame(data=clus_response_mask).to_csv(os.path.join(result_folder, "cluster_response_mask.csv"), index=False, header=False)
    # avg_FR_inh = np.array([shankA_inh,shankB_inh,shankC_inh,shankD_inh],dtype = float)
    # avg_FR_act = np.array([shankA_act,shankB_act,shankC_act,shankD_act],dtype = float)
   
    # savemat(os.path.join(result_folder, "population_stat_responsive_only.mat"), data_dict)
    np.save(os.path.join(result_folder,'all_clus_property.npy'),list_all_clus)
    np.save(os.path.join(result_folder,'all_clus_pca_preprocessed.npy'),scaled_data_list_new)
    np.save(os.path.join(result_folder_imp_clusters,'amplitude_hist.npy'),lst_amplitudes_all) # primarily used for representative examples
    np.save(os.path.join(result_folder_imp_clusters,'waveforms_all.npy'),lst_waveforms_all)     # primarily used for representative examples
    np.save(os.path.join(result_folder_imp_clusters,'clus_depth.npy'),lst_cluster_depth)    # primarily used for representative examples
    np.save(os.path.join(result_folder_imp_clusters,'ISI_hist_all.npy'),lst_isi_all)    # primarily used for representative examples


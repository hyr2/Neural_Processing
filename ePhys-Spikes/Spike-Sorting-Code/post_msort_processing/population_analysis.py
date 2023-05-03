import os, json
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
from Support import plot_all_trials, filterSignal_lowpass, filter_Savitzky_slow, filter_Savitzky_fast, zscore_bsl
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns

# Plotting fonts
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('font', size=16)          # controls default text sizes

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
    

def func_pop_analysis(session_folder,CHANNEL_MAP_FPATH):

    # Input parameters ---------------------
    # session folder is a single measurement 
    
    # firing rate calculation params
    WINDOW_LEN_IN_SEC = 40e-3
    SMOOTHING_SIZE = 11
    DURATION_OF_INTEREST = 2.5  # how many seconds to look at upon stim onset (this is the activation or inhibition window)
    # Setting up
    # session_path_str = "NVC/BC7/12-17-2021"
    # CHANNEL_MAP_FPATH = r"D:\Rice-Courses\neuroeng_lab\codes\stroke_proj_postproc\data_ch_maps\128chMap_flex.mat" # BC7
    # CHANNEL_MAP_FPATH = r"D:\Rice-Courses\neuroeng_lab\codes\stroke_proj_postproc\data_ch_maps\chan_map_1x32_128ch_rigid.mat" # BC6
    # CHANNEL_MAP_FPATH = r"D:\Rice-Courses\neuroeng_lab\codes\stroke_proj_postproc\data_ch_maps\Mirro_Oversampling_hippo_map.mat" # B-BC5
    # CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_rigid.mat'

    # session_folder = input('Input the source directory containing spike sorted and curated dataset for a single session:\n')
    # session_folder = '/home/hyr2-office/Documents/Data/NVC/RH-3/processed_data_rh3/10-19/'
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
    
    # Trial mask (only for newer animals)
    trial_mask = pd.read_csv(trial_mask_file, header=None, index_col=False,dtype = bool)
    TRIAL_KEEP_MASK = trial_mask[0].to_numpy(dtype = bool)
    TRIAL_KEEP_MASK = np.ones([Ntrials,],dtype = bool)

    # Channel mapping
    if (CHMAP2X16 == True):    # 2x16 channel map
        GH = 30
        GW_BWTWEENSHANKS = 250
    elif (CHMAP2X16 == False):  # 1x32 channel map
        GH = 25
        GW_BWTWEENSHANKS = 250

    # # Extracting data from summary file .xlsx
    # df_exp_summary = pd.read_excel(dir_expsummary)
    # arr_exp_summary = df_exp_summary.to_numpy()
    # Num_chan = arr_exp_summary[0,0]         # Number of channels
    # Notch_freq = arr_exp_summary[0,1]       # Notch frequencey selected (in Hz)
    # Fs = arr_exp_summary[0,2]               # Sampling freq (in Hz)
    # stim_start_time = arr_exp_summary[2,2]   # Stimulation start - 50ms of window
    # stim_start_time_original = arr_exp_summary[2,2]# original stimulation start time
    # n_stim_start = int(Fs * stim_start_time)# Stimulation start time in samples
    # Ntrials = arr_exp_summary[2,4]          # Number of trials
    # stim_end_time = arr_exp_summary[2,1] + stim_start_time  # End time of stimulation
    # time_seq = arr_exp_summary[2,0]         # Time of one sequence in seconds
    # Seq_perTrial =  arr_exp_summary[2,3]    # Number of sequences per trial
    # total_time = time_seq * Seq_perTrial    # Total time of the trial
    # print('Each sequence is: ', time_seq, 'sec')
    # time_seq = int(np.ceil(time_seq * Fs/2) * 2)                # Time of one sequence in samples (rounded up to even)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(result_folder_FR_avg):
        os.makedirs(result_folder_FR_avg)
    geom_path = os.path.join(session_folder, "geom.csv")
    # curation_mask_path = os.path.join(session_folder, "cluster_rejection_mask.npz")
    curation_mask_path = os.path.join(session_folder, 'accept_mask.csv')
    NATIVE_ORDERS = np.load(os.path.join(session_folder, "native_ch_order.npy"))
    axonal_mask = os.path.join(session_folder,'positive_mask.csv')

    # macro definitions
    ANALYSIS_NOCHANGE = 0       # A better name is non-stimulus locked
    ANALYSIS_EXCITATORY = 1     # A better name is activated
    ANALYSIS_INHIBITORY = -1    # A better name is suppressed

    # read cluster rejection data
    curation_masks = np.squeeze(pd.read_csv(curation_mask_path,header = None).to_numpy())
    single_unit_mask = curation_masks
    # single_unit_mask = curation_masks['single_unit_mask']
    # multi_unit_mask = curation_masks['multi_unit_mask']
    mask_axonal = np.squeeze(pd.read_csv(axonal_mask,header = None).to_numpy())  # axonal spikes
    mask_axonal = np.logical_not(mask_axonal)
    single_unit_mask = np.logical_and(mask_axonal,single_unit_mask)     # updated single unit mask
    
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

    trial_duration_in_samples = int(total_time*F_SAMPLE)
    window_in_samples = int(WINDOW_LEN_IN_SEC*F_SAMPLE)
    # read channel map
    geom = pd.read_csv(os.path.join(session_folder, "geom.csv"), header=None).values
    n_ch_this_session = geom.shape[0]
    print(geom.shape)
    # exit(0)
    # read trials times
    trials_start_times = loadmat(session_trialtimes)['t_trial_start'].squeeze()
    firings = readmda(os.path.join(session_folder, "firings.mda")).astype(np.int64)
    # get spike stamp for all clusters (in SAMPLEs not seconds)
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
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
        firing_rate_series = get_single_cluster_spikebincouts_all_trials(
            firing_stamp, 
            trials_start_times, 
            trial_duration_in_samples, 
            window_in_samples
            )
        firing_rate_series = firing_rate_series/WINDOW_LEN_IN_SEC       # This gives firing rate in Hz
        firing_rate_avg = np.mean(firing_rate_series[TRIAL_KEEP_MASK, :], axis=0) # averaging over all trials
        # firing_rate_avg = filterSignal_lowpass(firing_rate_avg, np.single(1/WINDOW_LEN_IN_SEC), axis_value = 0)
        firing_rate_avg = filter_Savitzky_slow(firing_rate_avg)
        firing_rate_sum = np.sum(firing_rate_series[TRIAL_KEEP_MASK, :], axis=0)
        
        n_samples_baseline = int(np.ceil(stim_start_time/WINDOW_LEN_IN_SEC))
        n_samples_stim = int(np.ceil((stim_end_time-stim_start_time)/WINDOW_LEN_IN_SEC))
        
        t_axis = np.linspace(0,firing_rate_avg.shape[0]*WINDOW_LEN_IN_SEC,firing_rate_avg.shape[0])
        t_1 = np.squeeze(np.where(t_axis >= 1.4))[0]      # stim start set to 2.45 seconds
        t_2 = np.squeeze(np.where(t_axis >= 2.0))[0]      # end of bsl region (actual value is 2.5s but FR starts to increase before. Maybe anticipatory spiking due to training)
        t_3 = np.squeeze(np.where(t_axis <= 2.4))[-1]        # stim end set to 5.15 seconds
        t_4 = np.squeeze(np.where(t_axis <= 4.5))[-1]
        t_5 = np.squeeze(np.where(t_axis <= 8))[-1]        # post stim quiet period
        t_6 = np.squeeze(np.where(t_axis <= 9.5))[-1]
        t_7 = np.squeeze(np.where(t_axis <= 4.5))[-1]
        t_8 = np.squeeze(np.where(t_axis <= 2.6))[-1]       # used for single trial
        t_9 = np.squeeze(np.where(t_axis <= 0.5))[-1]       # used for single trial
        # Zscore
        bsl_mean = np.mean(np.hstack((firing_rate_avg[t_1:t_2],firing_rate_avg[t_5:t_6]))) # baseline is pre and post stim
        bsl_std = np.std(np.hstack((firing_rate_avg[t_1:t_2],firing_rate_avg[t_5:t_6])))   # baseline is pre and post stim 
        firing_rate_zscore = zscore_bsl(firing_rate_avg, bsl_mean, bsl_std)          # Zscore to classify cluster as activated or suppressed
        
        # FR Classify
        (t_axis_s,firing_rate_zscore_s,normalized,indx) = FR_classifier_classic_zscore(firing_rate_zscore,t_axis,[t_1,t_7],[t_3,t_4])
        
        max_zscore_stim = np.amax(np.absolute(firing_rate_zscore[t_3:t_4]))          # Thresholding for Z score values
        # plt.figure()
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
        
        max_z = np.amax((firing_rate_zscore[t_3:t_4]))
        min_z = np.amin((firing_rate_zscore[t_3:t_4]))
        clus_property_1 = 0
        if max_zscore_stim > 2.5:
            if indx == 0 and N_spikes_local>(Ntrials*3) and (np.abs(max_z) > np.abs(min_z)):
                print('activated neuron')
                clus_property_1 = 1
            elif indx == 1 and N_spikes_local>(Ntrials*3) and (np.abs(max_z) < np.abs(min_z)):
                print('suppressed neuron')
                clus_property_1 = -1
        # print(clus_property)
        # if pval_2t > .01:
        #     clus_property = ANALYSIS_NOCHANGE
        # elif t_stat < 0:
        #     clus_property = ANALYSIS_EXCITATORY
        # else:
        #     clus_property = ANALYSIS_INHIBITORY
        # print(clus_property)
        
        # Computing number of spikes
        firing_rate_series = firing_rate_series * WINDOW_LEN_IN_SEC
        firing_rate_series_avg = np.sum(firing_rate_series,axis = 0)
        Spikes_stim = np.sum(firing_rate_series_avg[t_8:t_4])
        Spikes_bsl = np.sum(firing_rate_series_avg[t_9:t_3])
        Spikes_num = np.array([Spikes_bsl,Spikes_stim])
        

        return clus_property_1, firing_rate_avg, firing_rate_sum, Spikes_num


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
    for i_clus in range(n_clus):
        
        if single_unit_mask[i_clus] == True:
            # Need to add facilitating cell vs adapting cell vs no change cell
            clus_property, firing_rate_avg, firing_rate_sum, Spikes_num = single_cluster_main(i_clus)
            # creating a dictionary for this cluster
            firing_stamp = spike_times_by_clus[i_clus]
            N_spikes_local = spike_times_by_clus[i_clus].size 
            prim_ch = pri_ch_lut[i_clus]
            # Get shank ID from primary channel for the cluster
            shank_num = get_shanknum_from_msort_id(prim_ch)
            i_clus_dict  = {}
            i_clus_dict['cluster_id'] = i_clus + 1 # to align with the discard_noise_viz.py code cluster order [folder: figs_allclus_waveforms]
            i_clus_dict['total_spike_count'] = firing_stamp.shape
            i_clus_dict['prim_ch_coord'] = geom[prim_ch, :]
            i_clus_dict['shank_num'] = shank_num
            i_clus_dict['clus_prop'] = clus_property
            i_clus_dict['N_spikes'] = N_spikes_local
            i_clus_dict['N_spikes_stim'] =  Spikes_num[1]
            i_clus_dict['N_spikes_bsl'] = Spikes_num[0]
            list_all_clus.append(i_clus_dict)
            FR_series_all_clusters[iter_local].append(np.array(firing_rate_avg,dtype = float))
            # temporary
            # firing_rate_avg = 
            failed = (single_unit_mask[i_clus]==False)                       # SUA as well
            print(i_clus)
            if not(failed): 
                plot_all_trials(firing_rate_avg,1/WINDOW_LEN_IN_SEC,result_folder_FR_avg,i_clus_dict)           # Plotting function
            
            if clus_property==ANALYSIS_EXCITATORY:
                clus_response_mask[iter_local] = 1
            elif clus_property==ANALYSIS_INHIBITORY:
                clus_response_mask[iter_local] = -1
            else:
                clus_response_mask[iter_local] = 0
                
            if not(failed) and clus_property==ANALYSIS_EXCITATORY:          # Extracting FR of only activated neurons
                FR_list_byshank_act[shank_num].append(np.array(FR_series_all_clusters[iter_local],dtype = float))
            elif not(failed) and clus_property==ANALYSIS_INHIBITORY:
                FR_list_byshank_inh[shank_num].append(np.array(FR_series_all_clusters[iter_local],dtype = float))
            total_nclus_by_shank[shank_num] += 1
            if clus_property != ANALYSIS_NOCHANGE:
                # print("t=%.2f, p=%.4f"%(t_stat, pval_2t))
                if failed:
                    print("NOTE---- bad cluster with significant response to stim")
            if failed:
                continue
        
            
            if clus_property==ANALYSIS_EXCITATORY:
                act_nclus_by_shank[shank_num] += 1
            elif clus_property==ANALYSIS_INHIBITORY:
                inh_nclus_by_shank[shank_num] += 1
            else:
                nor_nclus_by_shank[shank_num] += 1
            
            if clus_property != ANALYSIS_NOCHANGE:
                if single_unit_mask[i_clus]:
                    single_nclus_by_shank[shank_num] += 1
                else:
                    multi_nclus_by_shank[shank_num] += 1
            iter_local = iter_local+1

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
    # compute # of channels per shank available
    num_channels_perShank = np.zeros([4,])
    for iter in NATIVE_ORDERS:
        print(iter)
        shanknum_local = get_shanknum_from_intan_id(iter)
        if shanknum_local == 0:
            num_channels_perShank[0] = num_channels_perShank[0] + 1
        elif shanknum_local == 1:
            num_channels_perShank[1] = num_channels_perShank[1] + 1
        elif shanknum_local == 2:
            num_channels_perShank[2] = num_channels_perShank[2] + 1
        elif shanknum_local == 3:
            num_channels_perShank[3] = num_channels_perShank[3] + 1
                
    
    # A mask for SUA and MUA
    FR_series_all_clusters = np.squeeze(np.array(FR_series_all_clusters))
    # accept_mask_local = np.logical_or(single_unit_mask == True, multi_unit_mask == True)
    # FR_series_all_clusters = FR_series_all_clusters[single_unit_mask]  # Only save the accepted clusters
    
    # Firing rate computation
    Time_vec = np.linspace(0,13.5,FR_series_all_clusters[1].shape[0])
    stim_start_idx = int(stim_start_time/WINDOW_LEN_IN_SEC)
    doi_end_idx = int((stim_start_time+DURATION_OF_INTEREST)/WINDOW_LEN_IN_SEC)
    stim_end_idx = int((stim_start_time+float(data_pre_ms['StimulationTime']))/WINDOW_LEN_IN_SEC)
    # activated neurons
    shankA_act = np.squeeze(FR_list_byshank_act[0])
    shankA_act = np.reshape(shankA_act,[int(shankA_act.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    shankB_act = np.squeeze(FR_list_byshank_act[1])
    shankB_act = np.reshape(shankB_act,[int(shankB_act.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    shankC_act = np.squeeze(FR_list_byshank_act[2])
    shankC_act = np.reshape(shankC_act,[int(shankC_act.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    shankD_act = np.squeeze(FR_list_byshank_act[3])
    shankD_act = np.reshape(shankD_act,[int(shankD_act.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
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
    shankA_act = np.mean(np.mean(shankA_act[:,stim_start_idx:doi_end_idx],axis=1)) if not np.isnan(shankA_act).any() else np.nan  # average FR during activation
    shankB_act = np.mean(np.mean(shankB_act[:,stim_start_idx:doi_end_idx],axis=1)) if not np.isnan(shankB_act).any()  else np.nan  # average FR during activation
    shankC_act = np.mean(np.mean(shankC_act[:,stim_start_idx:doi_end_idx],axis=1)) if not np.isnan(shankC_act).any()  else np.nan  # average FR during activation
    shankD_act = np.mean(np.mean(shankD_act[:,stim_start_idx:doi_end_idx],axis=1)) if not np.isnan(shankD_act).any()  else np.nan  # average FR during activation
    
    # supressed neurons
    # supressed neurons
    shankA_inh = np.squeeze(FR_list_byshank_inh[0])
    shankA_inh = np.reshape(shankA_inh,[int(shankA_inh.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    shankB_inh = np.squeeze(FR_list_byshank_inh[1])
    shankB_inh = np.reshape(shankB_inh,[int(shankB_inh.size/Time_vec.shape[0]),Time_vec.shape[0]])
    shankC_inh= np.squeeze(FR_list_byshank_inh[2])
    shankC_inh = np.reshape(shankC_inh,[int(shankC_inh.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    shankD_inh= np.squeeze(FR_list_byshank_inh[3])
    shankD_inh = np.reshape(shankD_inh,[int(shankD_inh.size/Time_vec.shape[0]),Time_vec.shape[0]]) 
    # compute bsl FR
    # shankA_inh_bsl = np.mean(shankA_inh[:,:stim_start_idx],axis = 1) if np.size(shankA_inh) != 0 else np.nan
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
    shankA_inh= np.mean(np.amin(shankA_inh[:,stim_start_idx:stim_end_idx], axis = 1)) if not np.isnan(shankA_inh).any() else np.nan  # average FR during activation
    shankB_inh= np.mean(np.amin(shankB_inh[:,stim_start_idx:stim_end_idx], axis = 1)) if not np.isnan(shankB_inh).any()  else np.nan  # average FR during activation
    shankC_inh= np.mean(np.amin(shankC_inh[:,stim_start_idx:stim_end_idx], axis = 1)) if not np.isnan(shankC_inh).any()  else np.nan  # average FR during activation
    shankD_inh= np.mean(np.amin(shankD_inh[:,stim_start_idx:stim_end_idx], axis = 1)) if not np.isnan(shankD_inh).any()  else np.nan  # average FR during activation

    # Output .mat file
    pd.DataFrame(data=clus_response_mask).to_csv(os.path.join(result_folder, "cluster_response_mask.csv"), index=False, header=False)
    avg_FR_inh = np.array([shankA_inh,shankB_inh,shankC_inh,shankD_inh],dtype = float)
    avg_FR_act = np.array([shankA_act,shankB_act,shankC_act,shankD_act],dtype = float)
    data_dict = {
        "total_nclus_by_shank": total_nclus_by_shank,
        "single_nclus_by_shank": single_nclus_by_shank,     # Number of activated/suppressed cluster (SUA)
        "multi_nclus_by_shank": multi_nclus_by_shank,       # Number of activated/suppressed cluster (MUA)
        "act_nclus_by_shank": act_nclus_by_shank,           # Number of activated cluster
        "inh_nclus_by_shank": inh_nclus_by_shank,           # Number of suppressed cluster
        "nor_nclus_by_shank": nor_nclus_by_shank,           # no response neurons 
        "FR_series_all_clusters": FR_series_all_clusters,   # FR series (Only accepted clusters and SUA)
        "avg_FR_act_by_shank": avg_FR_act,                  # Average over clusters of their "pk max firing rate" during stimulation
        "avg_FR_inh_by_shank": avg_FR_inh,                  # Average over clusters of their "pk min firing rate" during stimulation
        "numChan_perShank": num_channels_perShank           
    }
    savemat(os.path.join(result_folder, "population_stat_responsive_only.mat"), data_dict)
    np.save(os.path.join(result_folder,'all_clus_property.npy'),list_all_clus)
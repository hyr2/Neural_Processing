'''
Adapt to 2x16 channel maps
Non visualization yet
read Haad's rejection file -> update: simply reject first 6 trials
Author: Jia-ao
'''
import os, sys, json
sys.path.append(os.path.join(os.getcwd(),'utils-mountainsort'))
sys.path.append(os.getcwd())
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy.signal as signal
import pandas as pd
from utils.Support import read_stimtxt
from utils.read_mda import readmda
# from Support_MS import *

# from utils.read_stimtxt import read_stimtxt

# Files and Folders
source_dir = input('Input the source directory containing spike sorted and curated dataset for a single session:\n')
CHANNEL_MAP_FPATH = '/home/hyr2-office/Documents/git/Neural_SP/Neural_Processing/Channel_Maps/chan_map_1x32_128ch_rigid.mat'
stimtxt_path = os.path.join(source_dir,'whisker_stim.txt')
# session_folder = os.path.join(source_dir,'Processed','msorted')     # MS output
session_folder = source_dir
session_trialtimes = os.path.join(source_dir,'trials_times.mat')    # Trial times mat file
# manual_reject_fpath = os.path.join(session_folder, "manual_reject_clus.txt")
# geom_path = os.path.join(session_folder, "geom.csv")
curation_mask_path = os.path.join(session_folder, "cluster_rejection_mask.npz")
NATIVE_ORDERS = np.load(os.path.join(session_folder, "native_ch_order.npy"))
RESULT_PATH = os.path.join(source_dir,'Processed','firing_rates')
dir_expsummary = os.path.join(source_dir,'exp_summary.xlsx')
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
	
# Extracting data from summary file .xlsx
df_exp_summary = pd.read_excel(dir_expsummary)
arr_exp_summary = df_exp_summary.to_numpy()
Num_chan = arr_exp_summary[0,0]         # Number of channels
Notch_freq = arr_exp_summary[0,1]       # Notch frequencey selected (in Hz)
Fs = arr_exp_summary[0,2]               # Sampling freq (in Hz)
stim_start_time = arr_exp_summary[2,2]   # Stimulation start - 50ms of window
stim_start_time_original = arr_exp_summary[2,2]# original stimulation start time
n_stim_start = int(Fs * stim_start_time)# Stimulation start time in samples
Ntrials = arr_exp_summary[2,4]          # Number of trials
stim_end_time = arr_exp_summary[2,1] + stim_start_time  # End time of stimulation
time_seq = arr_exp_summary[2,0]         # Time of one sequence in seconds
Seq_perTrial =  arr_exp_summary[2,3]    # Number of sequences per trial
total_time = time_seq * Seq_perTrial    # Total time of the trial
print('Each sequence is: ', time_seq, 'sec')
time_seq = int(np.ceil(time_seq * Fs/2) * 2)                # Time of one sequence in samples (rounded up to even)
	
# Extract sampling frequency
file_pre_ms = os.path.join(source_dir,'pre_MS.json')
with open(file_pre_ms, 'r') as f:
  data_pre_ms = json.load(f)
F_SAMPLE = float(data_pre_ms['SampleRate'])
CHANNELMAP2X16 = bool(data_pre_ms['ELECTRODE_2X16'])      # this affects how the plots are generated

# --------------------- SET THESE PARAMETERS ------------------------------  
F_SAMPLE = Fs
WINDOW_LEN_IN_SEC = 10e-3
SMOOTHING_SIZE = 11
PLOT_SCALE_Y = True
DURATION_OF_INTEREST = 0.5  # how many seconds to look at upon stim onset
# chan_knob = 1

# Channel mapping
if (CHANNELMAP2X16 == True):    # 2x16 channel map
    GH = 30
    GW_BWTWEENSHANKS = 250
elif (CHANNELMAP2X16 == False):  # 1x32 channel map
    GH = 25
    GW_BWTWEENSHANKS = 250
    
chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']




# read cluster rejection data
single_unit_mask = np.load(curation_mask_path)['single_unit_mask']


# print(list(chmap_mat.keys()))
if np.min(chmap_mat)==1:
    print("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
    chmap_mat -= 1


# Some local functions used over and over again in the script: ----------------------------
def ch_convert_msort2intan(i_msort):
    """i_msort starts from 0; return index also starts from zero"""
    return NATIVE_ORDERS[i_msort]

def ch_convert_intan2msort(i_intan):
    """i_intan starts from 0; return msort index also starts from zero"""
    return np.where(NATIVE_ORDERS==i_intan)[0][0]

# def get_shanknum_from_intan_id(i_intan):
#     """i_intan is the native channel used in the .mat channel map starts from 0"""
#     return int(np.where(chmap_mat==i_intan)[1][0])

def get_shanknum_from_intan_id(i_intan):
    """FOR 2X16 MAPS: i_intan is the native channel used in the .mat channel map starts from 0"""
    return int(np.where(chmap_mat==i_intan)[1][0])//2

def get_shanknum_from_coordinate(x, y=None):
    "get shank number from coordinate"
    if isinstance(x, int):
        return int(x/GW_BWTWEENSHANKS)
    elif isinstance(x, np.ndarray) and x.shape==(2,):
        return int(x[0]/GW_BWTWEENSHANKS)
    else:
        raise ValueError("wrong input")

def get_intan_from_coordinate(x, y, CHANNELMAP2X16):
    "get intan index from coordinate"
    if (CHANNELMAP2X16 == True):
        return chmap_mat[int(y/GH), int(x/GW_BWTWEENSHANKS)*2+int((x%GW_BWTWEENSHANKS)>0)]
    elif (CHANNELMAP2X16 == False):  # 1x32 channel map
        return chmap_mat[int(y/GH), int(x/GW_BWTWEENSHANKS)]
# End of local functions -----------------------------------------------------------------


# Reading the whisker_stim.txt file 
STIM_START_TIME, stim_num, seq_period, len_trials, NUM_TRIALS, FramePerSeq, total_seq, len_trials_arr = read_stimtxt(stimtxt_path)
STIM_DURATION = stim_num * seq_period
TRIAL_DURATION = len_trials * seq_period


STIM_START_TIME_PLOT = STIM_START_TIME

# read trial rejection
TRIAL_KEEP_MASK = np.ones((NUM_TRIALS, ), dtype=bool)
print("Reject first 6 trials")
TRIAL_KEEP_MASK[:6] = False
# if os.path.exists(os.path.join(session_trialtimes, "reject.txt")):
#     print("We have trials to reject")
#     with open(os.path.join(session_trialtimes, "reject.txt")) as f:
#         rejected_trials = f.read().split('\n')[0]
#     rejected_trials = np.array([int(k)-1 for k in rejected_trials.split(' ')]) # start from zero in this code
#     TRIAL_KEEP_MASK[rejected_trials] = False

trial_duration_in_samples = int(TRIAL_DURATION*F_SAMPLE)
window_in_samples = int(WINDOW_LEN_IN_SEC*F_SAMPLE)

# read firing.mda
firings = readmda(os.path.join(session_folder, "firings.mda")).astype(np.int64)
# get spike stamp for all clusters (in SAMPLEs not seconds)
spike_times_all = firings[1,:]
spike_labels = firings[2,:]
n_clus = np.max(spike_labels)
print(n_clus)
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

# read channel map
geom = pd.read_csv(os.path.join(session_folder, "geom.csv"), header=None).values
n_ch_this_session = geom.shape[0]
print(geom.shape)
# exit(0)
# read trials times
trials_start_times = loadmat(session_trialtimes)['t_trial_start'].squeeze()


def process_single_cluster(firing_stamp, t_trial_start, trial_duration_in_samples, window_in_samples):
    # returns the spike bin counts; need to manuaaly divide result by window length to get real firing rate
    n_windows_in_trial = int(np.ceil(trial_duration_in_samples/window_in_samples))
    bin_edges = np.arange(0, window_in_samples*n_windows_in_trial+1, step=window_in_samples)
    n_trials = t_trial_start.shape[0]
    assert n_trials==NUM_TRIALS or n_trials==NUM_TRIALS+1, "%d %d" % (n_trials, NUM_TRIALS)
    if n_trials > NUM_TRIALS:
        n_trials = NUM_TRIALS
    firing_rate_series_by_trial = np.zeros((n_trials, n_windows_in_trial))
    for i in range(n_trials):
        trial_firing_mask = np.logical_and(firing_stamp>=t_trial_start[i], firing_stamp<=(t_trial_start[i]+trial_duration_in_samples))
        trial_firing_stamp = firing_stamp[trial_firing_mask] - t_trial_start[i]
        if trial_firing_stamp.shape[0]==0:
            continue
        tmp_hist, _ = np.histogram(trial_firing_stamp, bin_edges)
        firing_rate_series_by_trial[i,:] = tmp_hist
    return firing_rate_series_by_trial

firing_rates_by_channels = [[] for _ in range(n_ch_this_session)]

for i_clus in range(n_clus):
    if single_unit_mask[i_clus]==False:
        print("Cluster %d is rejected/multi-unit and skipped" %(i_clus+1))
        continue
    firing_stamp = spike_times_by_clus[i_clus]
    n_spikes = firing_stamp.shape[0]
    firing_rate_series_by_trial = process_single_cluster(firing_stamp, trials_start_times, trial_duration_in_samples, window_in_samples)
    firing_rate_series = np.mean(firing_rate_series_by_trial[TRIAL_KEEP_MASK, :], axis=0) # trial average
    firing_rate_series = firing_rate_series / WINDOW_LEN_IN_SEC # turn spike counts into firing rates
    prim_ch_this_session = pri_ch_lut[i_clus]
    smoother = signal.windows.hamming(SMOOTHING_SIZE)
    smoother = smoother / np.sum(smoother)
    firing_rate_series = signal.convolve(firing_rate_series, smoother, mode='same')
    firing_rates_by_channels[prim_ch_this_session].append(firing_rate_series)
    # print(i_clus)


valid_channel_ids_intan = []
valid_baseline_spike_rates = []
valid_normalized_spike_rate_series = []
peak_normalized_firing_rate_series = []
mean_normalized_firing_rate_series = []
area_under_normalized_curve_series = []
channel_ids_this_session = []

# count firing rate by channel
for i in range(n_ch_this_session):
    firing_rates_this_ch = firing_rates_by_channels[i]
    if len(firing_rates_this_ch) == 0:
        continue
    firing_rates_this_ch_agg = np.sum(np.vstack(firing_rates_this_ch), axis=0)
    stim_start_idx = int(STIM_START_TIME/WINDOW_LEN_IN_SEC)
    doi_end_idx = int((STIM_START_TIME+DURATION_OF_INTEREST)/WINDOW_LEN_IN_SEC)
    stim_end_idx = int((STIM_START_TIME+STIM_DURATION)/WINDOW_LEN_IN_SEC)
    baseline_firing_rate = np.mean(firing_rates_this_ch_agg[:stim_start_idx])
    # plt.plot(np.arange(firing_rates_this_ch_agg.shape[0])*WINDOW_LEN_IN_SEC, firing_rates_this_ch_agg, color='k')
    # plt.axvline(STIM_START_TIME, color='r')
    # plt.show()
    if baseline_firing_rate != 0:
        firing_rates_this_ch_agg = firing_rates_this_ch_agg/baseline_firing_rate - 1
    x, y = geom[i,:]
    # i_shank = int(x/GW) 
    prim_ch_intan = get_intan_from_coordinate(x,y,1) # the real index in the complete 128 channel map
    valid_channel_ids_intan.append(prim_ch_intan)
    valid_baseline_spike_rates.append(baseline_firing_rate)
    valid_normalized_spike_rate_series.append(firing_rates_this_ch_agg)
    channel_ids_this_session.append(i)
    peak_normalized_firing_rate_series.append(np.max(firing_rates_this_ch_agg[stim_start_idx:doi_end_idx]))
    mean_normalized_firing_rate_series.append(np.mean(firing_rates_this_ch_agg[stim_start_idx:doi_end_idx]))
    area_under_normalized_curve_series.append(WINDOW_LEN_IN_SEC*np.sum(np.abs(firing_rates_this_ch_agg[stim_start_idx:stim_end_idx])))
    

# first sort, then group channels by shank
ch_order_sorted = list(range(len(valid_channel_ids_intan)))
ch_order_sorted = sorted(ch_order_sorted, key=lambda idx: get_shanknum_from_intan_id(valid_channel_ids_intan[idx]))
valid_baseline_spike_rates_byshank = []
valid_channel_ids_intan_byshank = []
valid_normalized_spike_rate_series_byshank = []
peak_normalized_firing_rate_series_byshank = []
mean_normalized_firing_rate_series_byshank = []
area_under_normalized_curve_series_byshank = []
stim_locked_byshank = []
shanknums = []
key = lambda idx: get_shanknum_from_intan_id(valid_channel_ids_intan[idx])
for k,g in groupby(ch_order_sorted, key):
    group_this_shank = list(g)
    print("shanknum:%d; #channels recording valid single-unit clusters=%d" % (k, len(group_this_shank)))
    shanknums.append(k)
    valid_baseline_spike_rates_byshank.append([valid_baseline_spike_rates[i] for i in group_this_shank])
    valid_channel_ids_intan_byshank.append([valid_channel_ids_intan[i] for i in group_this_shank])
    valid_normalized_spike_rate_series_byshank.append([valid_normalized_spike_rate_series[i] for i in group_this_shank])
    peak_normalized_firing_rate_series_byshank.append([peak_normalized_firing_rate_series[i] for i in group_this_shank])
    mean_normalized_firing_rate_series_byshank.append([mean_normalized_firing_rate_series[i] for i in group_this_shank])
    area_under_normalized_curve_series_byshank.append([area_under_normalized_curve_series[i] for i in group_this_shank])
print("Shank numbers obtained by groupby:", shanknums)
# valid_channel_ids_intan_sorted = [valid_channel_ids_intan[i] for i in ch_order_sorted]
# valid_baseline_spike_rates_sorted = [valid_baseline_spike_rates[i] for i in ch_order_sorted]
# valid_normalized_spike_rate_series_sorted = [valid_normalized_spike_rate_series[i] for i in ch_order_sorted]
# peak_normalized_firing_rate_series_sorted = [peak_normalized_firing_rate_series[i] for i in ch_order_sorted]
# mean_normalized_firing_rate_series_sorted = [mean_normalized_firing_rate_series[i] for i in ch_order_sorted]



# save
##### !!!!! shanknums is a list of shanks where there are accepted clusters (starts from 0, e.g. shank 0 is shank A)
i_shank = -1
for i_shank_all in range(4):
    shanknum = "ABCD"[i_shank_all]
    if i_shank_all not in shanknums:
        print(shanknum, "Empty shank")
        matfile_dict = {
            "channel_ids_intan": [],
            "baseline_spike_rates": [],
            "normalized_spike_rate_series": [],
            "times_in_second": [],
            "peak_normalized_firing_rate_during_stim": [],
            "mean_normalized_firing_rate_during_stim": [],
            "area_under_normalized_curve_during_stim": [],
            "stim_locked": []
        }
        savemat(os.path.join(RESULT_PATH, "valid_normalized_spike_rates_by_channels_shank%s.mat"%(shanknum)), matfile_dict)
        continue
    i_shank += 1
    print(shanknum, i_shank)
    valid_baseline_spike_rates_byshank[i_shank] = np.array(valid_baseline_spike_rates_byshank[i_shank])
    valid_channel_ids_intan_byshank[i_shank] = np.array(valid_channel_ids_intan_byshank[i_shank])
    valid_normalized_spike_rate_series_byshank[i_shank] = np.vstack(valid_normalized_spike_rate_series_byshank[i_shank])
    peak_normalized_firing_rate_series_byshank[i_shank] = np.array(peak_normalized_firing_rate_series_byshank[i_shank])
    mean_normalized_firing_rate_series_byshank[i_shank] = np.array(mean_normalized_firing_rate_series_byshank[i_shank])
    stim_locked_byshank.append(np.absolute(mean_normalized_firing_rate_series_byshank[i_shank]) > 0.18)
    area_under_normalized_curve_series_byshank[i_shank] = np.array(area_under_normalized_curve_series_byshank[i_shank])
    print(valid_normalized_spike_rate_series_byshank[i_shank].shape)
    matfile_dict = {
        "channel_ids_intan": valid_channel_ids_intan_byshank[i_shank],
        "baseline_spike_rates": valid_baseline_spike_rates_byshank[i_shank],
        "normalized_spike_rate_series": valid_normalized_spike_rate_series_byshank[i_shank],
        "times_in_second": np.arange(valid_normalized_spike_rate_series_byshank[i_shank].shape[1])*WINDOW_LEN_IN_SEC,
        "peak_normalized_firing_rate_during_stim": peak_normalized_firing_rate_series_byshank[i_shank],
        "mean_normalized_firing_rate_during_stim": mean_normalized_firing_rate_series_byshank[i_shank],
        "area_under_normalized_curve_during_stim": area_under_normalized_curve_series_byshank[i_shank],
        "stim_locked": stim_locked_byshank[i_shank]
    }
    savemat(os.path.join(RESULT_PATH, "valid_normalized_spike_rates_by_channels_shank%s.mat"%(shanknum)), matfile_dict)

# exit(0)
valid_baseline_spike_rates = np.array(valid_baseline_spike_rates)
valid_channel_ids_intan = np.array(valid_channel_ids_intan)
valid_normalized_spike_rate_series = np.vstack(valid_normalized_spike_rate_series)
size_const=4
if PLOT_SCALE_Y:
    print(valid_normalized_spike_rate_series.shape)
    print()
    y_scale_max = np.max(valid_normalized_spike_rate_series[:, int(STIM_START_TIME/WINDOW_LEN_IN_SEC):int((STIM_START_TIME+STIM_DURATION)/WINDOW_LEN_IN_SEC)])
    y_scale_min = np.min(valid_normalized_spike_rate_series[:, int(STIM_START_TIME/WINDOW_LEN_IN_SEC):int((STIM_START_TIME+STIM_DURATION)/WINDOW_LEN_IN_SEC)])

fig = plt.figure(figsize=(32*size_const,4*size_const))
for i in range(valid_channel_ids_intan.shape[0]):
    prim_ch_intan = valid_channel_ids_intan[i]
    firing_rates_this_ch = valid_normalized_spike_rate_series[i]
    baseline_rate = valid_baseline_spike_rates[i]
    plot_trange = np.arange(firing_rates_this_ch.shape[0])*WINDOW_LEN_IN_SEC + WINDOW_LEN_IN_SEC/2
    i_msort = ch_convert_intan2msort(prim_ch_intan)
    x,y = geom[i_msort,0], geom[i_msort, 1]
    plot_row, plot_col = (31-int(y/25)), (int(x/300))
    plt.subplot(4,32, plot_col*32+plot_row+1)
    # plt.subplot(32, 4, plot_row*4+plot_col)
    plt.plot(plot_trange, firing_rates_this_ch, color='k')
    plt.axvline(STIM_START_TIME_PLOT, linestyle='--', color='r')
    plt.axvline(STIM_START_TIME_PLOT+STIM_DURATION, linestyle='--', color='r')
    # plt.text(0.9,0.9,"Ch%d; Baseline=%.3f spikes/sec"%(i, baseline_firing_rate))#, fontsize=15)
    if PLOT_SCALE_Y:
        plt.ylim(y_scale_min, y_scale_max)
    plt.title("Ch%d--%.2f"%(prim_ch_intan, baseline_rate))
    if plot_col<3:
        plt.xticks([], [])
    # if plot_row>0:
    #     plt.yticks([], [])
# plt.tight_layout()

if PLOT_SCALE_Y:
    plt.savefig(os.path.join(RESULT_PATH, "normalized_firing_rates_by_ch_yscaled.png"))
    plt.savefig(os.path.join(RESULT_PATH, "normalized_firing_rates_by_ch_yscaled.svg"))
    plt.savefig("tmp_yscaled.png")
else:
    plt.savefig(os.path.join(RESULT_PATH, "normalized_firing_rates_by_ch.png"))
    plt.savefig(os.path.join(RESULT_PATH, "normalized_firing_rates_by_ch.svg"))
    plt.savefig("tmp.png")
plt.close()
print("Done")



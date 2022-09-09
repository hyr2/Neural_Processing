import os 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy.signal as signal
import pandas as pd

from utils.read_mda import readmda
from utils.read_stimtxt import read_stimtxt


F_SAMPLE = 25e3
WINDOW_LEN_IN_SEC = 10e-3
SMOOTHING_SIZE = 11
session_path_str = "NVC/BC7/12-12-2021"
PLOT_SCALE_Y = False
CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat"
GH = 25
GW = 300


session_folder = os.path.join("/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/", session_path_str)
session_trialtimes = os.path.join("/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/trial_times/", session_path_str , "trials_times.mat")
manual_reject_fpath = os.path.join(session_folder, "manual_reject_clus.txt")
geom_path = os.path.join(session_folder, "geom.csv")
RESULT_PATH = os.path.join(session_folder, "firing_rate_by_channels")
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
# print(list(chmap_mat.keys()))
if np.min(chmap_mat)==1:
    print("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
    chmap_mat -= 1

TRIAL_DURATION, NUM_TRIALS, STIM_START_TIME, STIM_DURATION = read_stimtxt(os.path.join("/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data", session_path_str, "whisker_stim.txt"))
STIM_START_TIME_PLOT = STIM_START_TIME - 30e-3

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

# read manually rejected clusters
manual_rejected_mask = np.ones((n_clus,), dtype=bool)
try:
    with open(manual_reject_fpath, "r") as f:
        manual_rejected_clus = f.readlines()
        manual_rejected_clus = np.array([int(clus) for clus in manual_rejected_clus])
    manual_rejected_mask[manual_rejected_clus-1] = False
except:
    print("Manual rejection no done; plotting all")


def process_single_cluster(firing_stamp, t_trial_start, trial_duration_in_samples, window_in_samples):
    # returns the spike bin counts; need to manuaaly divide result by window length to get real firing rate
    n_windows_in_trial = int(np.ceil(trial_duration_in_samples/window_in_samples))
    bin_edges = np.arange(0, window_in_samples*n_windows_in_trial+1, step=window_in_samples)
    n_trials = t_trial_start.shape[0]
    assert n_trials==NUM_TRIALS, "%d %d" % (n_trials, NUM_TRIALS)
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
    if manual_rejected_mask[i_clus-1]==False:
        print("Cluster %d is rejected and skipped" %(i_clus+1))
        continue
    firing_stamp = spike_times_by_clus[i_clus]
    n_spikes = firing_stamp.shape[0]
    firing_rate_series_by_trial = process_single_cluster(firing_stamp, trials_start_times, trial_duration_in_samples, window_in_samples)
    firing_rate_series = np.mean(firing_rate_series_by_trial, axis=0) # trial average
    firing_rate_series = firing_rate_series / WINDOW_LEN_IN_SEC # turn spike counts into firing rates
    prim_ch_this_session = pri_ch_lut[i_clus]
    smoother = signal.windows.hamming(SMOOTHING_SIZE)
    smoother = smoother / np.sum(smoother)
    firing_rate_series = signal.convolve(firing_rate_series, smoother, mode='same')
    firing_rates_by_channels[prim_ch_this_session].append(firing_rate_series)
    # print(i_clus)

valid_channel_ids_from0 = []
valid_baseline_spike_rates = []
valid_normalized_spike_rate_series = []
channel_ids_this_session = []

for i in range(n_ch_this_session):
    firing_rates_this_ch = firing_rates_by_channels[i]
    if len(firing_rates_this_ch) == 0:
        continue
    firing_rates_this_ch_agg = np.sum(np.vstack(firing_rates_this_ch), axis=0)
    baseline_firing_rate = np.mean(firing_rates_this_ch_agg[:int(STIM_START_TIME/WINDOW_LEN_IN_SEC-1)])
    if baseline_firing_rate != 0:
        firing_rates_this_ch_agg = firing_rates_this_ch_agg/baseline_firing_rate - 1
    x, y = geom[i,:]
    prim_ch_real = chmap_mat[int(y/GH), int(x/GW)] # the real index in the complete 128 channel map
    valid_channel_ids_from0.append(prim_ch_real)
    valid_baseline_spike_rates.append(baseline_firing_rate)
    valid_normalized_spike_rate_series.append(firing_rates_this_ch_agg)
    channel_ids_this_session.append(i)

valid_baseline_spike_rates = np.array(valid_baseline_spike_rates)
valid_channel_ids_from0 = np.array(valid_channel_ids_from0)
valid_normalized_spike_rate_series = np.vstack(valid_normalized_spike_rate_series)
# save
matfile_dict = {
    "channel_ids_from0": valid_channel_ids_from0,
    "baseline_spike_rates": valid_baseline_spike_rates ,
    "normalized_spike_rate_series": valid_normalized_spike_rate_series,
    "WINDOW_LEN_IN_SEC": WINDOW_LEN_IN_SEC
}
savemat(os.path.join(RESULT_PATH, "valid_normalized_spike_rates_by_channels.mat"), matfile_dict)

size_const=4
if PLOT_SCALE_Y:
    print(valid_normalized_spike_rate_series.shape)
    print()
    y_scale_max = np.max(valid_normalized_spike_rate_series[:, int(STIM_START_TIME/WINDOW_LEN_IN_SEC):int((STIM_START_TIME+STIM_DURATION)/WINDOW_LEN_IN_SEC)])
    y_scale_min = np.min(valid_normalized_spike_rate_series[:, int(STIM_START_TIME/WINDOW_LEN_IN_SEC):int((STIM_START_TIME+STIM_DURATION)/WINDOW_LEN_IN_SEC)])

fig = plt.figure(figsize=(32*size_const,4*size_const))
for i in range(valid_channel_ids_from0.shape[0]):
    prim_ch_real = valid_channel_ids_from0[i]
    firing_rates_this_ch = valid_normalized_spike_rate_series[i]
    baseline_rate = valid_baseline_spike_rates[i]
    plot_trange = np.arange(firing_rates_this_ch.shape[0])*WINDOW_LEN_IN_SEC + WINDOW_LEN_IN_SEC/2
    x,y = geom[i,:]
    plot_row, plot_col = (31-int(y/25)), (int(x/300))
    plt.subplot(4,32, plot_col*32+plot_row+1)
    # plt.subplot(32, 4, plot_row*4+plot_col)
    plt.plot(plot_trange, firing_rates_this_ch, color='k')
    plt.axvline(STIM_START_TIME_PLOT, linestyle='--', color='r')
    plt.axvline(STIM_START_TIME_PLOT+STIM_DURATION, linestyle='--', color='r')
    # plt.text(0.9,0.9,"Ch%d; Baseline=%.3f spikes/sec"%(i, baseline_firing_rate))#, fontsize=15)
    if PLOT_SCALE_Y:
        plt.ylim(y_scale_min, y_scale_max)
    plt.title("Ch%d--%.2f"%(prim_ch_real, baseline_rate))
    if plot_col<3:
        plt.xticks([], [])
    if plot_row>0:
        plt.yticks([], [])
# plt.tight_layout()

if PLOT_SCALE_Y:
    plt.savefig(os.path.join(RESULT_PATH, "normalized_firing_rates_by_ch_yscaled.png"))
    plt.savefig(os.path.join(RESULT_PATH, "normalized_firing_rates_by_ch_yscaled.svg"))
    plt.savefig("tmp9_yscaled.png")
else:
    plt.savefig(os.path.join(RESULT_PATH, "normalized_firing_rates_by_ch.png"))
    plt.savefig(os.path.join(RESULT_PATH, "normalized_firing_rates_by_ch.svg"))
    plt.savefig("tmp9.png")
plt.close()
print("Done")



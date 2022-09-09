import os 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy.signal as signal
import pandas as pd

from utils.read_mda import readmda
from utils.read_stimtxt import read_stimtxt


F_SAMPLE = 25e3
session_path_str = "NVC/BC7/12-06-2021"
CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat"
GH = 25
GW = 300
CHAN_IDX = [13]
session_folder = os.path.join("/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/", session_path_str)
FIG_PATH = os.path.join(session_folder, "firing_rate_specific_channels")
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)

FIRING_PATH = os.path.join(session_folder, "firing_rate_by_channels")
TRIAL_DURATION, NUM_TRIALS, STIM_START_TIME, STIM_DURATION = read_stimtxt(os.path.join("/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data", session_path_str, "whisker_stim.txt"))
STIM_START_TIME_PLOT = STIM_START_TIME - 30e-3

ddict = loadmat(os.path.join(FIRING_PATH, "valid_normalized_spike_rates_by_channels.mat"))
channel_ids_from0 = ddict['channel_ids_from0'].squeeze()
baseline_spike_rates = ddict['baseline_spike_rates'].squeeze()
normalized_spike_rate_series = ddict['normalized_spike_rate_series']
WINDOW_LEN_IN_SEC = ddict['WINDOW_LEN_IN_SEC'][0,0]


for ch_idx in CHAN_IDX:
    i = np.where(channel_ids_from0==ch_idx)[0][0]
    print("i=",i)
    firing_rates_this_ch = normalized_spike_rate_series[i,:]
    baseline_rate = baseline_spike_rates[i]
    plot_trange = np.arange(firing_rates_this_ch.shape[0])*WINDOW_LEN_IN_SEC + WINDOW_LEN_IN_SEC/2
    fig = plt.figure(figsize=(10,4))
    plt.plot(plot_trange.squeeze(), firing_rates_this_ch, color='k')
    plt.axvline(STIM_START_TIME_PLOT, linestyle='--', color='r')
    plt.axvline(STIM_START_TIME_PLOT+STIM_DURATION, linestyle='--', color='r')
    plt.title("%s\nCh%d--Baseline firing rate=%.2f"%(session_path_str.replace("/", "_"), ch_idx, baseline_rate))
    # plt.tight_layout()
    plt.xlabel("Time (sec)")
    plt.ylabel("Relative change to baseline")
    plt.savefig(os.path.join(FIG_PATH, "single_%d.svg"%(ch_idx)))
    plt.savefig("single_%d_%s.svg"%(ch_idx, session_path_str.replace("/", "_")))
    plt.close()
    print("Done")



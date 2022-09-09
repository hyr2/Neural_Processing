import os

import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import ttest_ind
import pandas as pd

from utils.read_mda import readmda
from utils.read_stimtxt import read_stimtxt

CHMAP2X16 = False
F_SAMPLE = 25e3
WINDOW_LEN_IN_SEC = 10e-3
SMOOTHING_SIZE = 11
session_path_str = "NVC/BC7/12-06-2021"
# CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat"
CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/oversampling_palvo_flex_intanAdapterMap.mat"

DURATION_OF_INTEREST = 0.5 # how many seconds to look at upon stim onset
GH = 25
GW = 300
session_folder = os.path.join("/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/", session_path_str)
session_trialtimes = os.path.join("/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/trial_times/", session_path_str , "trials_times.mat")
# manual_reject_fpath = os.path.join(session_folder, "manual_reject_clus.txt")
geom_path = os.path.join(session_folder, "geom.csv")
curation_mask_path = os.path.join(session_folder, "cluster_rejection_mask.npz")
RESULT_PATH = os.path.join(session_folder, "firing_rate_by_channels_220111_refined_clusters")
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
# NATIVE_ORDERS = np.load(os.path.join(session_folder, "native_ch_order.npy"))

ANALYSIS_NOCHANGE = 0
ANALYSIS_EXCITATORY = 1
ANALYSIS_INHIBITORY = -1

# read cluster rejection data
curation_masks = np.load(curation_mask_path, allow_pickle=True)
single_unit_mask = curation_masks['single_unit_mask']
multi_unit_mask = curation_masks['multi_unit_mask']

chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
# print(list(chmap_mat.keys()))
if np.min(chmap_mat)==1:
    print("Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
    chmap_mat -= 1

# def ch_convert_msort2intan(i_msort):
#     """i_msort starts from 0; return index also starts from zero"""
#     # x, y = geom[i_msort,:]
#     # i_intan = chmap_mat[int(y/GH), int(x/GW)] # the real index in the complete 128 channel map
#     # return i_intan
#     return NATIVE_ORDERS[i_msort]

# def ch_convert_intan2msort(i_intan):
#     """i_intan starts from 0; return msort index also starts from zero"""
#     # pos = np.where(chmap_mat==i_intan)
#     # y_idx, x_idx = pos[0][0], pos[1][0]
#     # i_msort = np.where(np.logical_and(geom[:,0]==int(x_idx*GW), geom[:,1]==int(y_idx*GH)))
#     # return i_msort
#     return np.where(NATIVE_ORDERS==i_intan)[0][0]

def get_shanknum_from_intan_id(i_intan):
    """i_intan is the native channel used in the .mat channel map starts from 0"""
    if CHMAP2X16:
        return int(np.where(chmap_mat==i_intan)[1][0])//2
    else:
        return int(np.where(chmap_mat==i_intan)[1][0])
# def get_shanknum_from_intan_id(i_intan):
#     """FOR 2X16 MAPS: i_intan is the native channel used in the .mat channel map starts from 0"""
#     return int(np.where(chmap_mat==i_intan)[1][0])//2

# def get_shanknum_from_msort_id(i_msort):
#     return get_shanknum_from_intan_id(NATIVE_ORDERS[i_msort])
def get_shanknum_from_msort_id(i_msort):
    return get_shanknum_from_coordinate(int(geom[i_msort, 0]), int(geom[i_msort,1]))

def get_shanknum_from_coordinate(x, y=None):
    "get shank number from coordinate"
    if isinstance(x, int):
        return int(x/GW)
    elif isinstance(x, np.ndarray) and x.shape==(2,):
        return int(x[0]/GW)
    else:
        raise ValueError("wrong input")

TRIAL_DURATION, NUM_TRIALS, STIM_START_TIME, STIM_DURATION = read_stimtxt(os.path.join("/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data", session_path_str, "whisker_stim.txt"))
# STIM_START_TIME_PLOT = STIM_START_TIME - 30e-3

# read trial rejection
TRIAL_KEEP_MASK = np.ones((NUM_TRIALS, ), dtype=bool)
if os.path.exists(os.path.join(session_trialtimes, "reject.txt")):
    with open(os.path.join(session_trialtimes, "reject.txt")) as f:
        rejected_trials = f.read().split('\n')[0]
    rejected_trials = np.array([int(k)-1 for k in rejected_trials.split(' ')]) # start from zero in this code
    TRIAL_KEEP_MASK[rejected_trials] = False

trial_duration_in_samples = int(TRIAL_DURATION*F_SAMPLE)
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


def get_single_cluster_spikebincouts_all_trials(firing_stamp, t_trial_start, trial_duration_in_samples, window_in_samples):
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

def single_cluster_main(i_clus):
    
    firing_stamp = spike_times_by_clus[i_clus] 
    firing_rate_series = get_single_cluster_spikebincouts_all_trials(
        firing_stamp, 
        trials_start_times, 
        trial_duration_in_samples, 
        window_in_samples
        )
    firing_rate_avg = np.mean(firing_rate_series[TRIAL_KEEP_MASK, :], axis=0)
    n_samples_baseline = int(np.ceil(STIM_START_TIME/WINDOW_LEN_IN_SEC))
    n_samples_stim = int(np.ceil(STIM_DURATION/WINDOW_LEN_IN_SEC))
    t_stat, pval_2t = ttest_ind(
        firing_rate_avg[:n_samples_baseline], 
        firing_rate_avg[1+n_samples_baseline:1+n_samples_baseline+n_samples_stim], 
        equal_var=False)
    
    if pval_2t > .001:
        clus_property = ANALYSIS_NOCHANGE
    elif t_stat < 0:
        clus_property = ANALYSIS_EXCITATORY
    else:
        clus_property = ANALYSIS_INHIBITORY
    # print(clus_property)

    return (clus_property, t_stat, pval_2t), firing_rate_avg


total_nclus_by_shank = np.zeros(4)
single_nclus_by_shank = np.zeros(4)
multi_nclus_by_shank = np.zeros(4)
act_nclus_by_shank = np.zeros(4)
inh_nclus_by_shank = np.zeros(4)
nor_nclus_by_shank = np.zeros(4)

for i_clus in range(n_clus):
    (clus_property, t_stat, pval_2t), firing_rate_avg = single_cluster_main(i_clus)
    failed = (single_unit_mask[i_clus]==False and multi_unit_mask[i_clus]==False)
    if clus_property != ANALYSIS_NOCHANGE:
        print("t=%.2f, p=%.4f"%(t_stat, pval_2t))
        if failed:
            print("    NOTE---- bad cluster with significant response to stim")
    if failed:
        continue
    prim_ch = pri_ch_lut[i_clus]
    shank_num = get_shanknum_from_msort_id(prim_ch)
    if clus_property==ANALYSIS_EXCITATORY:
        act_nclus_by_shank[shank_num] += 1
    elif clus_property==ANALYSIS_INHIBITORY:
        inh_nclus_by_shank[shank_num] += 1
    else:
        nor_nclus_by_shank[shank_num] += 1
    
    if single_unit_mask[i_clus]:
        single_nclus_by_shank[shank_num] += 1
    else:
        multi_nclus_by_shank[shank_num] += 1

    total_nclus_by_shank[shank_num] += 1

data_dict = {
    "total_nclus_by_shank": total_nclus_by_shank,
    "single_nclus_by_shank": single_nclus_by_shank,
    "multi_nclus_by_shank": multi_nclus_by_shank,
    "act_nclus_by_shank": act_nclus_by_shank,
    "inh_nclus_by_shank": inh_nclus_by_shank, 
    "nor_nclus_by_shank": nor_nclus_by_shank
}

savemat(os.path.join(session_folder, "population_stat.mat"), data_dict)
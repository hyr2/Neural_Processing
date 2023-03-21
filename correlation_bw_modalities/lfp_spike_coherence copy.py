import os, sys
import json
from collections import OrderedDict
from itertools import groupby
import gc

import numpy as np
import scipy
import scipy.signal as signal
import scipy.fftpack as fftpack
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

import utils_cc
from utils_cc import read_stimtxt, readmda
sys.path.append("../ePhys-Spikes/Spike-Sorting-Code/preprocess_rhd/")
from utils.mdaio import DiskReadMda

def save_trial_ephys_resample(path_whole_mda, path_dest, *, trial_stamps, resample_factor=(1, 60), trial_duration, pad_samples, save=True, read_cache):
    # IF the Intan data was 30kHz, then a default resampling factor of (1,60) would resample to 500Hz
    up, down = resample_factor
    if (read_cache) and os.path.exists(path_dest):
        print("    In func save_trial_ephys_resample: %s alread exists. Reading it"%(path_dest))
        npz_temp = np.load(path_dest)
        if (npz_temp['resample_up']==up and npz_temp['resample_down']==down):
            return npz_temp
        else:
            del(npz_temp)
            gc.collect()

    reader = DiskReadMda(path_whole_mda)
    if pad_samples is None:
        ValueError("In func save_trial_ephys_resample: pad_samples must be provided")
    n_ch = reader.N1()
    trials_data = []
    pad_samples_re = int(pad_samples*up/down)
    
    for i_trial, start_stamp_sample in enumerate(trial_stamps):
        print("    In func save_trial_ephys_resample: trial#", i_trial)
        assert start_stamp_sample>=pad_samples and start_stamp_sample+trial_duration+pad_samples < reader.N2()
        # pad samples
        d_raw = reader.readChunk(i1=0, i2=start_stamp_sample-pad_samples, N1=n_ch, N2=trial_duration+2*pad_samples) # (should be n_channels x n_samples)
        if up != down:
            d_downsampled = signal.resample_poly(d_raw, up, down, axis=-1)
        else:
            d_downsampled = d_raw
        # d_downsampled = d_downsampled[:, pad_samples_re:-pad_samples_re]
        # d_raw = reader.readChunk(i1=0, i2=start_stamp_sample, N1=n_ch, N2=trial_duration) # (should be n_channels x n_samples)
        # d_downsampled = signal.resample_poly(d_raw, up, down, axis=-1)
        trials_data.append(d_downsampled)
    trials_data = np.stack(trials_data) # (n_trials, n_channels, n_samples_downsampled)
    trials_npz = dict(trials_data=trials_data, resample_up=up, resample_down=down, pad_samples_raw=pad_samples, pad_is_clipped=False)
    if save:
        np.savez(path_dest, **trials_npz)
    return trials_npz

def read_spiking(folderpath):
    """Given a session directory, obtain the spiking stamps for all the accepted units in seconds"""
    # TODO this function might eventually be replaced once we have clean MDA files
    with open(os.path.join(folderpath, "pre_MS.json"), "r") as f:
        session_info = json.load(f)
    fs_ephys = session_info["SampleRate"]
    # read spike stamps
    firings = readmda(os.path.join(folderpath, "firings.mda")).astype(int)
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
    n_clus = np.max(spike_labels)
    spike_times_by_clus =[[] for i in range(n_clus)]
    for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
        spike_times_by_clus[spk_lbl-1].append(spk_time-1)
    # organize spike stamps into list of ndarrays with time unit in "seconds"
    
    for i in range(n_clus):
        spike_times_by_clus[i] = np.array(spike_times_by_clus[i])/fs_ephys
    # mask units
    accept_mask = np.load(os.path.join(folderpath, "cluster_rejection_mask.npz"))["single_unit_mask"]
    # print("n_clus=",np.sum(accept_mask))
    spike_stamps = [stamp for i_unit, stamp in enumerate(spike_times_by_clus) if accept_mask[i_unit]]
    map_curation2msort = np.arange(n_clus)[accept_mask]
    unit_locs = pd.read_csv(os.path.join(folderpath, "clus_locations.csv"), header=None).to_numpy()
    unit_locs = unit_locs[accept_mask, :]
    return spike_stamps, map_curation2msort, unit_locs



def point_in_box(x,y,box):
    upperleft_x, upperleft_y, width, height = box
    # print(x, y, box)
    tmp_x = ((x >= upperleft_x) and (x < upperleft_x+width))
    tmp_y = ((y >= upperleft_y) and (y < upperleft_y+height))
    return (tmp_x and tmp_y)
    
def determine_region(point,boxes_list, boxes_dict):
    """ 
    boxes_list must be the keys of boxes_dict. This redundancy makes sure the result is correctly labelled and numbered
    (in case of unordered dictionaries)
    """
    for i_region, box_name in enumerate(boxes_list):
        if point_in_box(point[0], point[1], boxes_dict[box_name]):
            return i_region
    raise ValueError("Point (%.2f, %.2f) is not in any of the boxes"%(point[0], point[1]))

def coherence_one_session(session_spk_dir, session_lfp_dir):
    """
    Return a list of dicts; each dict is a trial-average cross-correlation result b/w some units' firing rates and some IOS ROI.
    """
    ret = []

    # read session and trials metadata
    with open(os.path.join(session_spk_dir, "pre_MS.json"), "r") as f:
        session_info = json.load(f)
    trial_duration = session_info["SequenceTime"]*session_info["SeqPerTrial"]
    raw_sfreq = session_info['SampleRate']
    trial_start_stamps = loadmat(os.path.join(session_spk_dir, "trials_times.mat"))['t_trial_start'].squeeze()
    n_trials = trial_start_stamps.shape[0]
    if os.path.exists(os.path.join(session_spk_dir, "trial_mask.csv")):
        trial_mask = pd.read_csv(os.path.join(session_spk_dir, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    else:
        trial_mask = np.ones(n_trials, dtype=bool)

    # read and resample LFP
    trial_duration_samples = int(session_info["SequenceTime"]*session_info["SeqPerTrial"]*session_info['SampleRate'])
    trial_start_stamps = loadmat(os.path.join(session_spk_dir, "trials_times.mat"))['t_trial_start'].squeeze() # in samples
    n_trials = trial_start_stamps.shape[0]
    if os.path.exists(os.path.join(session_spk_dir, "trial_mask.csv")):
        trial_mask = pd.read_csv(os.path.join(session_spk_dir, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    else:
        print("INFO: trial_mask not found!")
        trial_mask = np.ones(n_trials, dtype=bool)
    
    path_wholemda = os.path.join(session_lfp_dir, "converted_data.mda")
    path_dest = os.path.join(session_lfp_dir, "ephys_trial_padded_300hz.npz")
    trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 100), pad_samples=30000, save=True, read_cache=True) # 300Hz
    # data_r, bin_len = utils_cc.resample_0phase(data, session_info['SequenceTime'], 18, 5, axis=-1)
    # data_r = data_r - np.mean(data_r, axis=-1)[..., None] # MEAN SUBSTRACTION - THIS IS VERY IMPORTANT!!!
    trials_data = trials_npz['trials_data'] # (n_trials, n_chs, n_samples)
    up = trials_npz["resample_up"]
    down = trials_npz["resample_down"]
    ephys_newfreq = raw_sfreq*up/down
    if trials_npz["pad_is_clipped"]==False:
        padding_re = int(trials_npz["pad_samples_raw"]*up/down)
        trials_data = trials_data[:, :, padding_re:-padding_re]
    
    # band pass filter
    # band_edges = [30, 60]
    # sos = signal.butter(8, np.array(band_edges)/ephys_newfreq*2, btype="bandpass", output='sos')
    # trials_data = signal.sosfilt(sos, trials_data, axis=-1)
    # data_r = trials_data
    data_r = np.transpose(trials_data, [1,0,2]) # (n_chs, n_trials, n_samples)
    data_r = np.mean(data_r, axis=0)[None, ...]
    # trials_gamma_env = np.mean(trials_gamma_env, axis=0)[None, :] # average all the channels

    # read and prepare spiking time series
    bin_len = 1 / ephys_newfreq
    spike_stamps, map_curation2msort, unit_locs = read_spiking(session_spk_dir)
    # print(len(spike_stamps), len(unit_locs))
    # exit(0)
    trial_start_stamps_corrected = trial_start_stamps[trial_mask]/session_info['SampleRate'] + session_info['SequenceTime']/2 - bin_len/2
    trial_segments = [(tst, trial_duration) for tst in trial_start_stamps_corrected]
    spiking_timeseries = [utils_cc.bin_spikes(spk_stamp, trial_segments, bin_len) for spk_stamp in spike_stamps] # (n_units, n_trials, n_samples)
    # print([len(k) for k in spiking_timeseries[0]])
    spiking_timeseries = np.array(spiking_timeseries).astype(float)
    # print(spiking_timeseries.shape, spiking_timeseries.dtype)

    # TODO sum the spikes in the same approximate region (halfshank)
    # probes_region_offset_x = 50
    # if session_info["ELECTRODE_2X16"]:
    #     probes_region_width = 250
    #     probes_region_height = 30*15/2
    # else:
    #     probes_region_width = 300
    #     probes_region_height = 25*31/2
    
    
    # unit_regions_dict = OrderedDict( [
    #     # values: (box_upperleft_x, box_upperleft_y, box_width, box_height) // upper means smaller y; left means smaller x
    #     ("Atop", (0,0,probes_region_offset_x, probes_region_height)), # lambda x,y: ((x<probes_region_offset_x) and (y<probes_region_height)),
    #     ("Abot", (0,probes_region_height,probes_region_offset_x, probes_region_height)),
    #     ("Btop", (probes_region_width,0,probes_region_offset_x, probes_region_height)),
    #     ("Bbot", (probes_region_width,probes_region_height,probes_region_offset_x, probes_region_height)),
    #     ("Ctop", (2*probes_region_width,0,probes_region_offset_x, probes_region_height)),
    #     ("Cbot", (2*probes_region_width,probes_region_height,probes_region_offset_x, probes_region_height)),
    #     ("Dtop", (3*probes_region_width,0,probes_region_offset_x, probes_region_height)),
    #     ("Dbot", (3*probes_region_width,probes_region_height,probes_region_offset_x, probes_region_height)),
    # ])
    # unit_regions_list = list(unit_regions_dict.keys())

    # spiking_regions = [determine_region(point, unit_regions_list, unit_regions_dict) for point in unit_locs]
    
    # # reorder spiking timeseries by spiking regions
    # unit_ids_sorted = np.argsort(spiking_regions)
    # # spiking_rgn_ts_pair = [(spiking_regions[i_tmp], spiking_timeseries[i_tmp]) for i_tmp in unit_ids_sorted]
    # spiking_timeseries_groupavg = np.zeros((len(unit_regions_list), spiking_timeseries.shape[1], spiking_timeseries.shape[2]), dtype=float)
    # for spiking_rgn_idx, iterable_unit_ids in groupby(unit_ids_sorted, key=lambda i: spiking_regions[i]):
    #     # print(list(list_unit_ids))
    #     list_unit_ids = list(iterable_unit_ids)
    #     if len(list_unit_ids)==0:
    #         continue
    #     tmp_series = spiking_timeseries[list_unit_ids,:,:] # (n_units_in_region, n_trials, n_samples)
    #     spiking_timeseries_groupavg[spiking_rgn_idx, :, :] = np.sum(tmp_series, axis=0)

    # clus_id = 7
    # trial_id = 8
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(np.arange(spiking_timeseries[clus_id,trial_id,:].shape[-1])*bin_len, spiking_timeseries[clus_id,trial_id,:])
    
    # spiking_timeseries = utils_cc.hamming_smooth(spiking_timeseries_groupavg, 5, -1) # (n_regions, n_trials, n_samples)
    # spiking_timeseries = utils_cc.hamming_smooth(spiking_timeseries, 5, -1) # (n_regions, n_trials, n_samples)
    # plt.subplot(212)
    # plt.plot(np.arange(spiking_timeseries[clus_id,trial_id,:].shape[-1])*bin_len, spiking_timeseries[clus_id,trial_id,:])
    # plt.show()
    spiking_timeseries = spiking_timeseries - np.mean(spiking_timeseries, axis=-1)[..., None] # MEAN SUBSTRACTION - THIS IS VERY IMPORTANT!!!

    assert spiking_timeseries.shape[-2] == data_r.shape[-2], "#trials don't match"
    n_trials = data_r.shape[-2]

    n_samples = min(data_r.shape[-1], spiking_timeseries.shape[-1])
    if n_samples < data_r.shape[-1]:
        data_r = data_r[..., :n_samples]
    elif n_samples < spiking_timeseries.shape[-1]:
        # spiking_timeseries = spiking_timeseries[..., -1]
        spiking_timeseries = spiking_timeseries[..., :n_samples]


    # 
    # TODO vectorize
    # id_iosmod = 0 
    # for i_region in range(len(unit_regions_list)):
    #     if np.all(spiking_timeseries[i_region, :, :]==0):
    #         continue
    #     for id_roi in range(data_r.shape[0]):
    #         corr_all = []
    #         for i_trial in range(n_trials): 
    #             sig_a = data_r[id_roi, id_iosmod, i_trial, :]
    #             sig_b = spiking_timeseries[i_region, i_trial, :]
    #             # iterate over all trials
    #             corr_lags, corr_res = utils_cc.corr_normalized(sig_a, sig_b, sampling_interval=bin_len, unbiased=True, normalized=True)
    #             # corr_lags, corr_res = utils_cc.cosine_affinity_with_lags(sig_a, sig_b, sampling_interval=bin_len)
    #             # corrlag_samples_mask = (corr_lags>plot_range[0])&(corr_lags<plot_range[1])
    #             # plt.figure(figsize=(20, 12))
    #             # plt.subplot(211)
    #             # plt.plot(np.arange(sig_a.shape[0])*bin_len, sig_a, marker='.', color="blue")
    #             # ax1 = plt.gca()
    #             # ax1.set_xlabel("Time (seconds)")
    #             # ax1.set_ylabel("A - Imaging signal\n(resampled and mean subtracted)", color="blue")
    #             # ax2 = plt.gca().twinx()
    #             # ax2.plot(np.arange(sig_b.shape[0])*bin_len, sig_b, marker='.', color="orange")
    #             # ax2.set_ylabel("B - Binned firing counts\n(smoothed and mean subtracted)", color='orange')
    #             # plt.subplot(212)
    #             # plt.plot(corr_lags[corrlag_samples_mask], corr_res[corrlag_samples_mask], linewidth=0.7, marker='x', label='Cross-correlation')
    #             # plt.xlabel("Lag (seconds) (Negative means A is earlier)")
    #             # plt.legend()
    #             # plt.show()
    #             corr_all.append(corr_res)
    #         corr_all = np.array(corr_all)
    #         corr_avg = np.mean(corr_all, axis=0)
    #         corr_std = np.std(corr_all, axis=0)
    #         temp_res = {}
    #         temp_res["description"]="IOS-MOD%d-ROI%d-vs-unitsInShank%s" % (id_iosmod, id_roi, unit_regions_list[i_region])
    #         temp_res["corr_avg"] = corr_avg
    #         temp_res["corr_std"] = corr_std
    #         temp_res["corr_lags"] = corr_lags
    #         ret.append(temp_res)
    #         print(temp_res["description"])

    # id_iosmod = 0 
    n_fft = n_samples//2 + 1# ((n_samples/2+1) if (n_samples%2==0) else ((n_samples+1)/2))
    coherences = np.empty((spiking_timeseries.shape[0], data_r.shape[0], n_fft), dtype=float)
    rfftfreq = np.arange(n_fft) / (bin_len*n_samples)
    for i_unit in range(spiking_timeseries.shape[0]):
        for i_ch in range(data_r.shape[0]):
            sig_a_by_trial = data_r[i_ch, :, :]
            sig_b_by_trial = spiking_timeseries[i_unit, :, :]
            spectra_a = scipy.fft.rfft(sig_a_by_trial, axis=-1) # (n_trials, n_fft)
            spectra_b = scipy.fft.rfft(sig_b_by_trial, axis=-1)
            crospectrum_ab_avg = np.abs(np.mean(spectra_a*np.conj(spectra_b), axis=0))
            sqrspectrum_a_avg = np.mean(np.real(spectra_a*np.conj(spectra_a)), axis=0)
            sqrspectrum_b_avg = np.mean(np.real(spectra_b*np.conj(spectra_b)), axis=0)
            coherences[i_unit, i_ch, :] = crospectrum_ab_avg/(np.sqrt(sqrspectrum_a_avg*sqrspectrum_b_avg))
    return coherences, rfftfreq

if __name__ == "__main__":
    # ios_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/IOS/processed_data_bc7/"
    spk_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_bc7/"
    lfp_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/mytempfolder/BC7"

    fig_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/bc7_211206_lfp_spk_coherence_fs300hz_230308"
    session_rel_dirs = ["2021-12-06"] #os.listdir(spk_dir)


    best_lag_datasets = []
    best_score_datasets = []
    all_corrs_datasets = []

    for session_rel_dir in session_rel_dirs:
        print(session_rel_dir)
        session_spk_dir = os.path.join(spk_dir, session_rel_dir)
        session_lfp_dir = os.path.join(lfp_dir, session_rel_dir)
        session_fig_dir = os.path.join(fig_dir, session_rel_dir)
        if not os.path.exists(session_fig_dir):
            os.makedirs(session_fig_dir)
        coherences, rfftfreq = coherence_one_session(session_spk_dir, session_lfp_dir)
        coherences = signal.savgol_filter(coherences, 51, 3, axis=-1)
        coherences = coherences[..., ::20]
        rfftfreq = rfftfreq[::20]
        for i_unit in range(coherences.shape[0]):
            for i_ch in range(coherences.shape[1]):
                plt.figure()
                plt.bar(rfftfreq, coherences[i_unit, i_ch, :], width=rfftfreq[1]-rfftfreq[0])
                # plt.fill_between(corr_lags_valid, corr_avg_valid-corr_std_valid, corr_avg_valid+corr_std_valid, color='orange', alpha=0.3)
                # plt.ylim([0,1])
                plt.axhline(0, color='r', linestyle="--")
                plt.axvline(0, color='yellow', linestyle="--")
                plt.xlabel("Freq (Hz)")
                plt.title("Coherence - spiking unit %d -vs- LFP ch %d " % (i_unit, i_ch))
                # plt.show()
                plt.savefig(os.path.join(session_fig_dir,"SPC_U%dC%d.png") % (i_unit, i_ch))
                plt.close()
    exit()
    days = [-5, -3, -1, 2, 7, 14, 21, 28, 35, 42]
    plt.figure()
    plt.boxplot(best_lag_datasets, positions=days, showfliers=False, widths=3)
    for day, ds in zip(days, best_lag_datasets):
        plt.scatter([day]*len(ds), ds, color='k')
    # plt.bar(days, best_lags[kna], color='k', width=0.3)
    plt.xticks(days)
    plt.title("LFP -vs- spikes - Best Lag")
    plt.xlim([-8, 45])
    plt.savefig(os.path.join(fig_dir, "spike-vs-lfp_bestlag.png"))
    plt.close()

    plt.figure()
    plt.boxplot(best_score_datasets, positions=days, showfliers=False, widths=3)
    for day, ds in zip(days, best_score_datasets):
        plt.scatter([day]*len(ds), ds, color='k')
    # plt.bar(days, best_lags[kna], color='k', width=0.3)
    plt.xticks(days)
    plt.title("LFP -vs- spikes - Best Score")
    plt.xlim([-8, 45])
    plt.savefig(os.path.join(fig_dir, "spike-vs-LFP_bestscore.png"))
    plt.close()

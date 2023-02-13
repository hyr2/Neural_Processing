import os
import json
from collections import OrderedDict
from itertools import groupby

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

import utils_cc
from utils_cc import read_stimtxt, readmda



def read_ios_signals(folderpath):
    x = []
    x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial1.mat")))
    x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial2.mat")))
    x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial3.mat")))
    x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial4.mat")))
    x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial5.mat")))
    x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial6.mat")))
    x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial7.mat")))
    trial_mask = pd.read_csv(os.path.join(folderpath, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    # print(trial_mask.shape)
    data = np.stack([xx['ios_trials_local'][trial_mask, :, :] for xx in x], axis=0) # (nROIs, nTrials, time, n_modalities)
    data = np.transpose(data, axes=[0,3,1,2]) # (nROIs, nModalities, nTrials, nTime)
    return data

def corrs_one_session(session_dir):
    """
    Return a list of dicts; each dict is a trial-average cross-correlation result b/w some units' firing rates and some IOS ROI.
    """
    ret = []
    # read session and trials metadata
    # trial_mask = pd.read_csv(os.path.join(session_dir, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    with open(os.path.join(session_dir, "pre_MS.json"), "r") as f:
        session_info = json.load(f)
    # trial_duration = session_info["SequenceTime"]*session_info["SeqPerTrial"]
    # trial_start_stamps = loadmat(os.path.join(session_dir, "trials_times.mat"))['t_trial_start'].squeeze()

    # read and resample Imaging signal
    data = read_ios_signals(session_dir)  # (nROIs, nModalities, nTrials, nTime)
    data_r, bin_len = utils_cc.resample_0phase(data, session_info['SequenceTime'], 18, 5, axis=-1)
    data_r = data_r - np.mean(data_r, axis=-1)[..., None] # MEAN SUBSTRACTION - THIS IS VERY IMPORTANT!!!
    n_trials = data_r.shape[2]
    id_iosmod = 0 
    for i_roi in [0,1,2,4,5]:# range(data_r.shape[0]-1):
        for j_roi in [6]:# range(i_roi+1, data_r.shape[0]):
            corr_all = []
            for i_trial in range(n_trials): 
                sig_a = data_r[i_roi, id_iosmod, i_trial, :]
                sig_b = data_r[j_roi, id_iosmod, i_trial, :]
                # iterate over all trials
                corr_lags, corr_res = utils_cc.corr_normalized(sig_a, sig_b, sampling_interval=bin_len, unbiased=True, normalized=True)
                corr_all.append(corr_res)
            corr_all = np.array(corr_all)
            corr_avg = np.mean(corr_all, axis=0)
            corr_std = np.std(corr_all, axis=0)
            temp_res = {}
            temp_res["description"]="IOS-MOD%d-ROI%d-vs-ROI%d" % (id_iosmod, i_roi, j_roi)
            temp_res["corr_avg"] = corr_avg
            temp_res["corr_std"] = corr_std
            temp_res["corr_lags"] = corr_lags
            ret.append(temp_res)
            print(temp_res["description"])
    return ret

if __name__ == "__main__":
    # figdirs = [
    #     "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/rh8_imaging/2022-12-01",
    #     "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/rh8_imaging/2022-12-03",
    #     "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/rh8_imaging/2022-12-05"
    #     ]
    figdir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/rh8_imaging_vein_all"
    session_dirs=[
        "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_rh8/2022-12-01",
        "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_rh8/2022-12-03",
        "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_rh8/2022-12-05"
        ]

    best_lags = {}
    best_scores = {}
    all_corrs = {}
    for session_dir in session_dirs:
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        ret = corrs_one_session(session_dir)
        plot_range = [-3.5, 3.5]
        for d in ret:
            corr_lags = d["corr_lags"]
            corr_avg = d["corr_avg"]
            corr_std = d["corr_std"]
            corrlag_samples_mask = (corr_lags>plot_range[0])&(corr_lags<plot_range[1])
            print(np.sum(corrlag_samples_mask))
            corr_avg_valid = corr_avg[corrlag_samples_mask]
            corr_std_valid = corr_std[corrlag_samples_mask]
            corr_lags_valid = corr_lags[corrlag_samples_mask]
            best_lag_idx = np.argmax(np.abs(corr_avg_valid))
            best_lag = corr_lags_valid[best_lag_idx]
            best_score = corr_avg_valid[best_lag_idx]
            kname = d["description"]
            if kname not in best_lags.keys():
                best_lags[kname] = [best_lag]
                best_scores[kname] = [best_score]
                all_corrs[kname] = [(corr_lags_valid, corr_avg_valid)]
            else:
                best_lags[kname].append(best_lag)
                best_scores[kname].append(best_score)
                all_corrs[kname].append(tuple([corr_lags_valid, corr_avg_valid]))
            # plt.figure()
            # plt.plot(corr_lags_valid, corr_avg_valid, linewidth=0.7, marker='x', label='mean correlation')
            # plt.fill_between(corr_lags_valid, corr_avg_valid-corr_std_valid, corr_avg_valid+corr_std_valid, color='orange', alpha=0.3)
            # plt.ylim([-1,1])
            # plt.axhline(0, color='r', linestyle="--")
            # plt.axvline(0, color='yellow', linestyle="--")
            # plt.xlabel("Lag (seconds) (Negative means A is earlier)")
            # plt.title(d["description"])
            # plt.show()
            # plt.savefig(os.path.join(figdir, d["description"]+".png"))
            # plt.close()
    days = [0,1,2]
    for kname in best_lags.keys():
        plt.figure()
        plt.bar(days, best_lags[kname], color='k', width=0.3)
        plt.xticks(days)
        plt.title(kname+" - Best Lag")
        plt.savefig(os.path.join(figdir, kname+"_bestlag.png"))
        plt.close()

        plt.figure()
        plt.bar(days, best_scores[kname], color='k', width=0.3)
        plt.xticks(days)
        plt.title(kname+" - Best Correlation Score")
        plt.savefig(os.path.join(figdir, kname+"_bestscore.png"))
        plt.close()

        plt.figure()
        for i in range(len(all_corrs[kname])):
            plt.plot(all_corrs[kname][i][0], all_corrs[kname][i][1])
        plt.title(kname + " - Cross correlation")
        plt.savefig(os.path.join(figdir, kname+"_corrs.png"))
        plt.close()








# print(corr_avg_valid.shape, corr_lags_valid.shape)

# 

# smoothed_3 = utils_cc.box_smooth(spiking_timeseries, 5, -1)
# clus_id = 7
# trial_id = 8
# plt.figure()
# plt.subplot(411)
# plt.plot(np.arange(n_samples)*bin_len, spiking_timeseries[clus_id,trial_id,:])
# plt.subplot(412)
# plt.plot(np.arange(n_samples)*bin_len, smoothed_1[clus_id,trial_id,:])
# # plt.subplot(413)
# # plt.plot(np.arange(n_samples)*bin_len, smoothed_2[clus_id,trial_id,:])
# plt.subplot(414)
# plt.plot(np.arange(n_samples)*bin_len, smoothed_3[clus_id,trial_id,:])
# plt.show()

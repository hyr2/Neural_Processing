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

def read_lfp_data(session_spk_dir, session_lfp_path):

    with open(os.path.join(session_spk_dir, "pre_MS.json"), "r") as f:
        session_info = json.load(f)
    raw_sfreq = session_info['SampleRate']
    trial_start_stamps = loadmat(os.path.join(session_spk_dir, "trials_times.mat"))['t_trial_start'].squeeze() # in samples
    n_trials = trial_start_stamps.shape[0]
    if os.path.exists(os.path.join(session_spk_dir, "trial_mask.csv")):
        print("  TRIAL MASK EXISTS AT: %s"%(os.path.join(session_spk_dir, "trial_mask.csv")))
        trial_mask = pd.read_csv(os.path.join(session_spk_dir, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    else:
        print("  TRIAL MASK NONEXISTANT: %s" % (os.path.join(session_spk_dir, "trial_mask.csv")))
        trial_mask = np.ones(n_trials, dtype=bool)

    trials_npz = np.load(session_lfp_path)
    trials_data = trials_npz['trials_data'] # (n_trials, n_chs, n_samples)
    up = trials_npz["resample_up"]
    down = trials_npz["resample_down"]
    
    ephys_newfreq = raw_sfreq*up/down
    trials_data = trials_data[trial_mask, :, :]
    

    h = signal.firwin(513, np.array([30, 100])/ephys_newfreq*2, window="hamming", pass_zero=False)
    trials_data_gamma = signal.filtfilt(h, 1, trials_data, axis=-1)
    trials_data_gamma_env = np.abs(signal.hilbert(trials_data_gamma, axis=-1))
    if trials_npz["pad_is_clipped"]==False:
        padding_re = int(trials_npz["pad_samples_raw"]*up/down)
        trials_data = trials_data[:, :, padding_re:-padding_re]
        trials_data_gamma = trials_data_gamma[:, :, padding_re:-padding_re]
        trials_data_gamma_env = trials_data_gamma_env[:, :, padding_re:-padding_re]
    return trials_data_gamma_env, ephys_newfreq

def read_ios_signals(session_spk_dir, session_ios_dir):
    x = []
    x.append(loadmat(os.path.join(session_ios_dir, "Processed/mat_files", "IOS_singleTrial1.mat")))
    x.append(loadmat(os.path.join(session_ios_dir, "Processed/mat_files", "IOS_singleTrial2.mat")))
    x.append(loadmat(os.path.join(session_ios_dir, "Processed/mat_files", "IOS_singleTrial3.mat")))
    x.append(loadmat(os.path.join(session_ios_dir, "Processed/mat_files", "IOS_singleTrial4.mat")))
    x.append(loadmat(os.path.join(session_ios_dir, "Processed/mat_files", "IOS_singleTrial5.mat")))
    x.append(loadmat(os.path.join(session_ios_dir, "Processed/mat_files", "IOS_singleTrial6.mat")))
    x.append(loadmat(os.path.join(session_ios_dir, "Processed/mat_files", "IOS_singleTrial7.mat")))
    if os.path.exists(os.path.join(session_spk_dir, "trial_mask.csv")):
        print("  TRIAL MASK EXISTS AT: %s"%(os.path.join(session_spk_dir, "trial_mask.csv")))
        trial_mask = pd.read_csv(os.path.join(session_spk_dir, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    else:
        print("  TRIAL MASK NONEXISTANT: %s" % (os.path.join(session_spk_dir, "trial_mask.csv")))
        trial_mask = np.ones(x[0]['ios_trials_local'].shape[0], dtype=bool)
    # print(trial_mask.shape)
    data = np.stack([xx['ios_trials_local'][trial_mask, :, :] for xx in x], axis=0) # (nROIs, nTrials, time, n_modalities)
    data = np.transpose(data, axes=[0,3,1,2]) # (nROIs, nModalities, nTrials, nTime)
    return data

def corrs_one_session(session_spk_dir, session_ios_dir, session_lfp_dir):
    """
    Return a list of dicts; each dict is a trial-average cross-correlation result b/w some units' firing rates and some IOS ROI.
    """
    ret = []
    # read session and trials metadata
    # trial_mask = pd.read_csv(os.path.join(session_dir, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    with open(os.path.join(session_spk_dir, "pre_MS.json"), "r") as f:
        session_info = json.load(f)
    # trial_duration = session_info["SequenceTime"]*session_info["SeqPerTrial"]
    # trial_start_stamps = loadmat(os.path.join(session_dir, "trials_times.mat"))['t_trial_start'].squeeze()

    # ephys_tempsampfreq should be 500Hz. THis code assumes that.
    trials_gamma_env, ephys_tempsampfreq = read_lfp_data(session_spk_dir, os.path.join(session_lfp_dir, "ephys_trial_padded.npz"))
    # trials_gamma_env = np.mean(trials_gamma_env, axis=1) # average all the channels
    trials_gamma_env = np.transpose(trials_gamma_env, [1,0,2]) # (n_chs, n_trials, n_samples)
    # trials_gamma_env = np.mean(trials_gamma_env, axis=0)[None, :] # average all the channels
    trials_gamma_env_r, bin_len0 = utils_cc.resample_0phase(trials_gamma_env, 1/ephys_tempsampfreq, 1, 50, axis=-1) # 100ms
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(trials_gamma_env[0,0,:])
    # plt.subplot(212)
    # plt.plot(trials_gamma_env_r[0,0,:])
    # plt.show()
    trials_gamma_env_r = trials_gamma_env_r - np.mean(trials_gamma_env_r, axis=-1)[..., None]
    # n_samples_lfp = trials_gamma_env_r.shape[-1]
    # read and resample Imaging signal
    data = read_ios_signals(session_spk_dir, session_ios_dir)  # (nROIs, nModalities, nTrials, nTime)
    data_r, bin_len = utils_cc.resample_0phase(data, session_info['SequenceTime'], 18, 10, axis=-1) # resampling period is 100ms
    data_r = data_r - np.mean(data_r, axis=-1)[..., None] # MEAN SUBSTRACTION - THIS IS VERY IMPORTANT!!!
    # n_trials = data_r.shape[2]
    # n_samples_ios = data_r.shape[-1]

    print("re-sampling interval of IOS and LFP signals:", bin_len, bin_len0)
    assert trials_gamma_env_r.shape[-2] == data_r.shape[-2], "#trials don't match : (%d, %d)"%(trials_gamma_env_r.shape[-2], data_r.shape[-2])
    n_trials = data_r.shape[-2]

    n_samples = min(data_r.shape[-1], trials_gamma_env_r.shape[-1])
    if n_samples < data_r.shape[-1]:
        data_r = data_r[..., :n_samples]
    elif n_samples < trials_gamma_env_r.shape[-1]:
        trials_gamma_env_r = trials_gamma_env_r[..., :n_samples]

    id_iosmod = 0 
    for i_roi in range(data_r.shape[0]):
        for i_ch in range(trials_gamma_env.shape[0]):
            corr_all = []
            for i_trial in range(n_trials): 
                sig_a = data_r[i_roi, id_iosmod, i_trial, :]
                sig_b = trials_gamma_env_r[i_ch, i_trial, :]
                # iterate over all trials
                corr_lags, corr_res = utils_cc.corr_normalized(sig_a, sig_b, sampling_interval=bin_len, unbiased=True, normalized=True)
                # plt.figure(figsize=(20, 12))
                # plt.subplot(211)
                # plt.plot(np.arange(sig_a.shape[0])*bin_len, sig_a, marker='.', color="blue")
                # ax1 = plt.gca()
                # ax1.set_xlabel("Time (seconds)")
                # ax1.set_ylabel("A - Imaging signal\n(resampled and mean subtracted)", color="blue")
                # ax2 = plt.gca().twinx()
                # ax2.plot(np.arange(sig_b.shape[0])*bin_len, sig_b, marker='.', color="orange")
                # ax2.set_ylabel("B - LFP Gamma Envelope\n(resampled and mean subtracted)", color='orange')
                # plt.subplot(212)
                # plt.plot(corr_lags, corr_res, linewidth=0.7, marker='x', label='Cross-correlation')
                # plt.xlabel("Lag (seconds) (Negative means A is earlier)")
                # plt.legend()
                # plt.show()
                corr_all.append(corr_res)
            corr_all = np.array(corr_all)
            corr_avg = np.mean(corr_all, axis=0)
            corr_std = np.std(corr_all, axis=0)
            temp_res = {}
            temp_res["description"]="IOS-MOD%d-LFPiintan%d-vs-ROI%d" % (id_iosmod, i_ch, i_roi)
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
    ios_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/IOS/processed_data_rh8/"
    spk_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_rh8/"
    lfp_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/mytempfolder/RH-8"

    fig_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/rh8_ios-lfp-avg-230227"
    session_rel_dirs= ["2022-12-03"] # os.listdir(spk_dir)


    best_lags = {}
    best_scores = {}
    all_corrs = {}
    for session_rel_dir in session_rel_dirs:
        # if not os.path.exists(figdir):
        #     os.makedirs(figdir)
        session_spk_dir = os.path.join(spk_dir, session_rel_dir)
        session_ios_dir = os.path.join(ios_dir, session_rel_dir)
        session_lfp_dir = os.path.join(lfp_dir, session_rel_dir)
        session_fig_dir = os.path.join(fig_dir, session_rel_dir)
        if not os.path.exists(session_fig_dir):
            os.makedirs(session_fig_dir)
        ret = corrs_one_session(session_spk_dir, session_ios_dir, session_lfp_dir)
        plot_range = [-8, 8]
        for d in ret:
            corr_lags = d["corr_lags"]
            corr_avg = d["corr_avg"]
            corr_std = d["corr_std"]
            corrlag_samples_mask = (corr_lags>plot_range[0])&(corr_lags<plot_range[1])
            # print(np.sum(corrlag_samples_mask))
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
            plt.figure()
            plt.plot(corr_lags_valid, corr_avg_valid, linewidth=0.7, marker='x', label='mean correlation')
            plt.fill_between(corr_lags_valid, corr_avg_valid-corr_std_valid, corr_avg_valid+corr_std_valid, color='orange', alpha=0.3)
            plt.ylim([-1,1])
            plt.axhline(0, color='r', linestyle="--")
            plt.axvline(0, color='yellow', linestyle="--")
            plt.xlabel("Lag (seconds) (Negative means A is earlier)")
            plt.title(d["description"])
            # plt.show()
            plt.savefig(os.path.join(session_fig_dir, d["description"]+".png"))
            plt.close()
    days = [-5, -3, -1, 2, 7, 14, 21, 28, 35, 42]
    for kname in best_lags.keys():
        plt.figure()
        plt.bar(days, best_lags[kname], color='k', width=0.3)
        plt.xticks(days)
        plt.title(kname+" - Best Lag")
        plt.savefig(os.path.join(fig_dir, kname+"_bestlag.png"))
        plt.close()

        plt.figure()
        plt.bar(days, best_scores[kname], color='k', width=0.3)
        plt.xticks(days)
        plt.title(kname+" - Best Correlation Score")
        plt.savefig(os.path.join(fig_dir, kname+"_bestscore.png"))
        plt.close()

        plt.figure()
        for i in range(len(all_corrs[kname])):
            plt.plot(all_corrs[kname][i][0], all_corrs[kname][i][1])
        plt.title(kname + " - Cross correlation")
        plt.savefig(os.path.join(fig_dir, kname+"_corrs.png"))
        plt.close()

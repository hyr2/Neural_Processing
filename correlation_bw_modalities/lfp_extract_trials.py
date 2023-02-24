# 1 read RHD and save as MDA for accessing arbitrary segment
#     (TODO figure out a way to do downsampling by chunk while keeping the sample alignment)
# 2 get trial stamps and read corresponding segments of ePhys
# 3 downsample, filter, Hilbert.
# 4 cross-correlation with spike

import os, sys
from copy import deepcopy
import gc
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.signal as signal
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

sys.path.append("../ePhys-Spikes/Spike-Sorting-Code/preprocess_rhd/")
from utils.mdaio import DiskReadMda


def save_trial_ephys_resample(path_whole_mda, path_dest, *, trial_stamps, resample_factor=(1, 60), trial_duration, pad_samples=30000):
    reader = DiskReadMda(path_whole_mda)
    n_ch = reader.N1()
    trials_data = []
    up, down = resample_factor
    pad_samples_re = int(pad_samples*up/down)
    for i_trial, start_stamp_sample in enumerate(trial_stamps):
        print(i_trial)
        assert start_stamp_sample>=pad_samples and start_stamp_sample+trial_duration+pad_samples < reader.N2()
        # pad samples
        d_raw = reader.readChunk(i1=0, i2=start_stamp_sample-pad_samples, N1=n_ch, N2=trial_duration+2*pad_samples) # (should be n_channels x n_samples)
        d_downsampled = signal.resample_poly(d_raw, up, down, axis=-1)
        # d_downsampled = d_downsampled[:, pad_samples_re:-pad_samples_re]
        # d_raw = reader.readChunk(i1=0, i2=start_stamp_sample, N1=n_ch, N2=trial_duration) # (should be n_channels x n_samples)
        # d_downsampled = signal.resample_poly(d_raw, up, down, axis=-1)
        trials_data.append(d_downsampled)
    trials_data = np.stack(trials_data) # (n_trials, n_channels, n_samples_downsampled)
    np.savez(path_dest, trials_data=trials_data, resample_up=up, resample_down=down, pad_samples_raw=pad_samples, pad_is_clipped=False)
    

if __name__ == "__main__":
    session_spk_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_rh8/2022-12-03"
    outputdir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/mytempfolder/"
    rel_dir = "RH-8/2022-12-03"
    fig_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/rh8_lfp_test"
    path_wholemda = os.path.join(outputdir, rel_dir, "converted_data.mda")
    path_dest = os.path.join(outputdir, rel_dir, "ephys_trial_padded.npz")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    with open(os.path.join(session_spk_dir, "pre_MS.json"), "r") as f:
        session_info = json.load(f)
    raw_sfreq = session_info['SampleRate']
    trial_duration_samples = int(session_info["SequenceTime"]*session_info["SeqPerTrial"]*raw_sfreq)
    trial_start_stamps = loadmat(os.path.join(session_spk_dir, "trials_times.mat"))['t_trial_start'].squeeze() # in samples
    n_trials = trial_start_stamps.shape[0]
    if os.path.exists(os.path.join(session_spk_dir, "trial_mask.csv")):
        trial_mask = pd.read_csv(os.path.join(session_spk_dir, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    else:
        trial_mask = np.ones(n_trials, dtype=bool)
    
    # save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples)

    trials_npz = np.load(path_dest)
    trials_data = trials_npz['trials_data'] # (n_trials, n_chs, n_samples)
    up = trials_npz["resample_up"]
    down = trials_npz["resample_down"]
    
    ephys_newfreq = raw_sfreq*up/down
    trials_data = trials_data[trial_mask, :, :]
    

    # h = signal.firwin(513, np.array([30, 100])/ephys_newfreq*2, window="hamming", pass_zero=False)
    # trials_data_gamma = signal.filtfilt(h, 1, trials_data, axis=-1)
    sos = signal.butter(8, np.array([30, 100])/ephys_newfreq*2, btype="bandpass", output='sos')
    trials_data_gamma = signal.sosfilt(sos, trials_data, axis=-1)
    trials_data_gamma_env = np.abs(signal.hilbert(trials_data_gamma, axis=-1))
    if trials_npz["pad_is_clipped"]==False:
        padding_re = int(trials_npz["pad_samples_raw"]*up/down)
        trials_data = trials_data[:, :, padding_re:-padding_re]
        trials_data_gamma = trials_data_gamma[:, :, padding_re:-padding_re]
        trials_data_gamma_env = trials_data_gamma_env[:, :, padding_re:-padding_re]
    n_samples = trials_data.shape[2]
    n_trials  = trials_data.shape[0]
    print(n_samples , n_samples / ephys_newfreq)
    trials_data = np.mean(trials_data, axis=1) # channel average -> (n_trials, n_samples)
    trials_data_gamma = np.mean(trials_data_gamma, axis=1) # channel average -> (n_trials, n_samples)
    trials_data_gamma_env = np.mean(trials_data_gamma_env, axis=1) # channel average -> (n_trials, n_samples)

    plt.figure(figsize=(10,10))
    plt.subplot(311)
    # plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data[0,0,:])
    plt.plot(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data, axis=0), color='k')
    plt.xticks([])
    plt.subplot(312)
    # plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data_gamma[0,0,:])
    plt.plot(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_gamma, axis=0), color='k')
    plt.fill_between(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_gamma, axis=0)-np.std(trials_data_gamma, axis=0),  np.mean(trials_data_gamma, axis=0)+np.std(trials_data_gamma, axis=0), color='gray', alpha=0.3)
    plt.xticks([])
    plt.subplot(313)
    # plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data_gamma[0,0,:])
    plt.plot(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_gamma_env, axis=0), color='k')
    plt.fill_between(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_gamma_env, axis=0)-np.std(trials_data_gamma_env, axis=0),  np.mean(trials_data_gamma_env, axis=0)+np.std(trials_data_gamma_env, axis=0), color='gray', alpha=0.3)
    # plt.xticks([])
    plt.savefig(os.path.join(fig_dir, "temp.png"))
    plt.show()
    plt.close()

    plt.figure(figsize=(10,10))
    f, t, Sxx = signal.spectrogram(trials_data_gamma, nperseg=64, noverlap=16, fs=ephys_newfreq, axis=-1)
    # for i in range(n_trials):
    #     plt.subplot(8,8, i+1)
    #     plt.gca().pcolor(t, f, Sxx[i, ...], cmap='hot')
    plt.pcolormesh(t, f, np.mean(Sxx, axis=0), cmap='hot')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(t, np.sum(np.mean(Sxx, axis=0), axis=0))
    plt.show()
    # plt.figure(figsize=(10,10))
    # plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data.T+np.arange(n_trials)*4*np.std(trials_data), linewidth=0.9)
    # plt.show()

    # plt.figure(figsize=(10,10))
    # plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data_gamma.T+np.arange(n_trials)*4*np.std(trials_data_gamma), linewidth=0.9)
    # plt.show()

    # plt.figure()
    # data_fft = np.abs(fftpack.fft(trials_data_gamma, axis=-1))
    # freqs = fftpack.fftfreq(n_samples, 1/ephys_newfreq)
    # plt.plot(freqs, data_fft.T+np.arange(n_trials)*4*np.std(data_fft), linewidth=0.9)
    # plt.show()
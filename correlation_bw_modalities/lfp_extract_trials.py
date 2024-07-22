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
from scipy.stats import zscore
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    session_spk_dir = "/home/hyr2-office/Documents/TransferToPC/LFP_CSD"
    outputdir = session_spk_dir
    rel_dir = "BC7/2021-12-06"
    fig_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/bc7_lfp_test_230301-dn500hz_30-100hz-2021-12-06"
    path_wholemda = os.path.join(outputdir, rel_dir, "converted_data.mda")
    path_dest = os.path.join(outputdir, rel_dir, "ephys_trial_padded_500hz.npz")
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
        print("INFO: trial_mask not found!")
        trial_mask = np.ones(n_trials, dtype=bool)
    
    
    trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 60), pad_samples=30000, save=True, read_cache=True) # 500Hz
    # trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 15), pad_samples=30000, save=False, read_cache=True) # 2000Hz
    # trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 1), pad_samples=30000, save=False, read_cache=False) # no resampling

    # trials_npz = np.load(path_dest)
    trials_data = trials_npz['trials_data'] # (n_trials, n_chs, n_samples)
    up = trials_npz["resample_up"]
    down = trials_npz["resample_down"]

    for i_trial in range(n_trials):
        if np.max(trials_data[i_trial, :, :] > 900):
            trial_mask[i_trial] = False
    
    ephys_newfreq = raw_sfreq*up/down
    trials_data = trials_data[trial_mask, :, :]
    

    # h = signal.firwin(513, np.array([30, 100])/ephys_newfreq*2, window="hamming", pass_zero=False)
    # trials_data_bpf = signal.filtfilt(h, 1, trials_data, axis=-1)
    band_edges = [30, 100]
    sos = signal.butter(8, np.array(band_edges)/ephys_newfreq*2, btype="bandpass", output='sos')
    trials_data_bpf = signal.sosfilt(sos, trials_data, axis=-1)
    # trials_data_bpf = trials_data
    trials_data_bpf_hil = signal.hilbert(trials_data_bpf, axis=-1)
    trials_data_bpf_env = np.abs(trials_data_bpf_hil)
    trials_data_bpf_phi = np.angle(trials_data_bpf_hil)
    if trials_npz["pad_is_clipped"]==False:
        padding_re = int(trials_npz["pad_samples_raw"]*up/down)
        trials_data = trials_data[:, :, padding_re:-padding_re]
        trials_data_bpf = trials_data_bpf[:, :, padding_re:-padding_re]
        trials_data_bpf_env = trials_data_bpf_env[:, :, padding_re:-padding_re]
        trials_data_bpf_phi = trials_data_bpf_phi[:, :, padding_re:-padding_re]
    n_samples = trials_data.shape[2]
    n_trials  = trials_data.shape[0]
    print(n_samples , n_samples / ephys_newfreq)
    f, t, Sxx = signal.spectrogram(trials_data_bpf, nperseg=256, noverlap=192, fs=ephys_newfreq, axis=-1)
    Sxx = zscore(Sxx, axis=-1)
    # # trials_data
    # trials_data = np.mean(trials_data, axis=1) # channel average -> (n_trials, n_samples)
    # trials_data_bpf = np.mean(trials_data_bpf, axis=1) # channel average -> (n_trials, n_samples)
    # trials_data_bpf_env = np.mean(trials_data_bpf_env, axis=1) # channel average -> (n_trials, n_samples)
    
    # data_1ch1trial = trials_data[0,0,:]
    # data_bpf_1ch1trial = 
    i_ch = 0
    # for i_trial in range(trials_data.shape[0]):
    #     print("#trial:", i_trial)
    #     plt.figure(figsize=(10,10))
    #     plt.subplot(311)
    #     plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data[i_trial,i_ch,:], color='k')
    #     plt.title("Raw data - channel#%d; trial#%d"%(i_ch, i_trial))
    #     plt.xticks([])
    #     plt.subplot(312)
    #     plt.title("Filtered data (%d - %d Hz) - one channel one trial"% (band_edges[0], band_edges[1]))
    #     plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data_bpf_env[i_trial, i_ch, :], color='orange')
    #     plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data_bpf[i_trial,i_ch,:], color='k')
    #     plt.xticks([])
    #     plt.subplot(313)
    #     plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data_bpf_phi[i_trial,i_ch,:], color='k')
    #     plt.title("Filtered phase (%d - %d Hz) - one channel one trial"% (band_edges[0], band_edges[1]))
    #     plt.savefig(os.path.join(fig_dir, "raw_filt_env_%d-%dhz_ch%d_tr%d.png"%(band_edges[0], band_edges[1], i_ch, i_trial)))
    #     # plt.show()
    #     plt.close()

    #     plt.figure(figsize=(10,10))
        
    #     # for i in range(n_trials):
    #     #     plt.subplot(8,8, i+1)
    #     #     plt.gca().pcolor(t, f, Sxx[i, ...], cmap='hot')
    #     plt.pcolormesh(t, f, Sxx[i_trial, i_ch, :, :], cmap='hot', shading="auto")
    #     plt.title("Spectrogram of filtered signal - channel#%d; trial#%d"%(i_ch, i_trial))
    #     plt.colorbar()
    #     plt.savefig(os.path.join(fig_dir, "spectrogram_%d-%dhz_ch%d_tr%d.png"%(band_edges[0], band_edges[1], i_ch, i_trial)))
    #     plt.close()

    #     plt.figure(figsize=(10,5))
    #     plt.title("Power of trial-averaged filtered signal (%d-%d Hz)- channel#%d; trial#%d"%(band_edges[0], band_edges[1], i_ch, i_trial))
    #     plt.plot(t, np.sum(Sxx[i_trial, i_ch, :, :], axis=0), color='k')
    #     # plt.show()
    #     plt.savefig(os.path.join(fig_dir, "spectrogram_band_power_%d-%dhz_ch%d_tr%d.png"%(band_edges[0], band_edges[1], i_ch, i_trial)))
    #     plt.close()

    avg_env = np.abs(signal.hilbert(np.mean(trials_data_bpf[:, i_ch, :], axis=0)))
    print("Doing trial average")
    plt.figure(figsize=(10,10))
    plt.subplot(311)
    plt.plot(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data[:,i_ch,:], axis=0), color='k')
    plt.title("Raw data - channel#%d; trial-avg"%(i_ch))
    plt.xticks([])
    plt.subplot(312)
    plt.plot(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_bpf[:,i_ch,:], axis=0), color='k')
    # plt.fill_between(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_bpf, axis=0)-np.std(trials_data_bpf, axis=0),  np.mean(trials_data_bpf, axis=0)+np.std(trials_data_bpf, axis=0), color='gray', alpha=0.3)
    # plt.xticks([])
    # plt.subplot(313)
    # plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data_bpf[0,0,:])
    plt.plot(np.arange(n_samples)/ephys_newfreq, avg_env, color='orange')
    i_mark = int(2*ephys_newfreq)
    onset = (i_mark + np.argmax(np.diff(avg_env[i_mark:int(3*ephys_newfreq)])))/ephys_newfreq
    plt.axvline(onset, color='red', linestyle='-.')
    plt.title("Filtered data (%d - %d Hz) - one channel trial avg: onset at %.4f sec"% (band_edges[0], band_edges[1], onset))
    # plt.plot(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_bpf_env[:, i_ch, :], axis=0), color='orange')
    # plt.plot(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_bpf_env, axis=0), color='orange')
    # plt.fill_between(np.arange(n_samples)/ephys_newfreq, np.mean(trials_data_bpf_env, axis=0)-np.std(trials_data_bpf_env, axis=0),  np.mean(trials_data_bpf_env, axis=0)+np.std(trials_data_bpf_env, axis=0), color='gray', alpha=0.3)
    # plt.title("Hilbert envelope - trial averaged")
    plt.xticks([])
    plt.subplot(313)
    plt.plot(np.arange(n_samples)/ephys_newfreq,  np.angle(signal.hilbert(np.mean(trials_data_bpf[:, i_ch, :], axis=0))), color='k')
    plt.title("Filtered phase (%d - %d Hz) - one channel otrial avg"% (band_edges[0], band_edges[1]))
    plt.savefig(os.path.join(fig_dir, "raw_filt_env_%d-%dhz_ch%d_tr-avg.png"%(band_edges[0], band_edges[1], i_ch)))
    # plt.show()
    plt.close()

    plt.figure(figsize=(10,10))
    
    # for i in range(n_trials):
    #     plt.subplot(8,8, i+1)
    #     plt.gca().pcolor(t, f, Sxx[i, ...], cmap='hot')
    plt.pcolormesh(t, f, np.mean(Sxx[:, i_ch, :, :], axis=0), cmap='hot', shading="auto")
    plt.title("Spectrogram of filtered signal - channel#%d; trial-avg"%(i_ch))
    plt.colorbar()
    plt.savefig(os.path.join(fig_dir, "spectrogram_%d-%dhz_ch%d_tr-avg.png"%(band_edges[0], band_edges[1], i_ch)))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.title("Power of trial-averaged filtered signal (%d-%d Hz)- channel#%d; trial-avg"%(band_edges[0], band_edges[1], i_ch))
    plt.plot(t, np.mean(np.sum(Sxx[:, i_ch, :, :], axis=1), axis=0), color='k')
    # plt.show()
    plt.savefig(os.path.join(fig_dir, "spectrogram_band_power_%d-%dhz_ch%d_tr-avg.png"%(band_edges[0], band_edges[1], i_ch)))
    plt.close()

        # plt.figure(figsize=(10,10))
        # plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data.T+np.arange(n_trials)*4*np.std(trials_data), linewidth=0.9)
        # plt.show()

        # plt.figure(figsize=(10,10))
        # plt.plot(np.arange(n_samples)/ephys_newfreq, trials_data_bpf.T+np.arange(n_trials)*4*np.std(trials_data_bpf), linewidth=0.9)
        # plt.show()

        # plt.figure()
        # data_fft = np.abs(fftpack.fft(trials_data_bpf, axis=-1))
        # freqs = fftpack.fftfreq(n_samples, 1/ephys_newfreq)
        # plt.plot(freqs, data_fft.T+np.arange(n_trials)*4*np.std(data_fft), linewidth=0.9)
        # plt.show()
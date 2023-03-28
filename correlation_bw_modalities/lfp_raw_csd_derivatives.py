"""
Compute CSD by smoothing the data and then (1) calculating 2nd order derivative or (2) using inverse-matrix method 
TODO modularize
"""

import os, sys
from copy import deepcopy
import gc
import json
from collections import OrderedDict
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import utils_cc
from utils_cc import inverse_csd, sort_and_groupby, reject_trials_by_ephys
sys.path.append("../ePhys-Spikes/Spike-Sorting-Code/preprocess_rhd/")
from utils.mdaio import DiskReadMda

def save_trial_ephys_resample(path_whole_mda, path_dest, *, trial_stamps, resample_factor=(1, 60), trial_duration, pad_samples, save, read_cache):
    # IF the Intan data was 30kHz, then a default resampling factor of (1,60) would resample to 500Hz
    up, down = resample_factor
    if (read_cache) and os.path.exists(path_dest):
        print("    In func save_trial_ephys_resample: %s alread exists. Reading it"%(path_dest))
        npz_temp = np.load(path_dest)
        if (npz_temp['resample_up']==up and npz_temp['resample_down']==down):
            return npz_temp
        else:
            print("     In func save_trial_ephys_resample: resampling factor does not match. Will read and create data")
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
        if not (start_stamp_sample>=pad_samples and start_stamp_sample+trial_duration+pad_samples < reader.N2()):
            warnings.warn("Won't be able to get enough out-of-trial samples for padding. Be aware of potential edge effect")
        # pad samples
        sample_start = max(0, start_stamp_sample-pad_samples)
        # print(reader.N2())
        sample_durat = min(trial_duration+2*pad_samples, reader.N2()-sample_start)
        # print(sample_durat)
        d_raw = reader.readChunk(i1=0, i2=sample_start, N1=n_ch, N2=sample_durat) # (should be n_channels x n_samples)
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

def interpolate_one_shank(this_shank_lfp_, i_depths_existing_, i_depths_to_interp_, desired_depths_, interp_mode_):
    print("DEBUG: interpolating from %d samples" % (len(i_depths_existing_)))
    below = this_shank_lfp_[i_depths_existing_[0], :]
    above = this_shank_lfp_[i_depths_existing_[-1], :]
    interpolator = interp1d(i_depths_existing_, this_shank_lfp_[i_depths_existing_, :], axis=0, kind=interp_mode_, assume_sorted=True, bounds_error=False, fill_value=(below, above))
    # print("Existing:", i_depths_existing)
    # print("To interpolate", i_depths_to_interp)
    if np.max(i_depths_to_interp_)>np.max(i_depths_existing_):
        print("WARNING: extrapolation occurs: extrapolating for depth=%.2f when max existing depth=%.2f"
            %(desired_depths_[np.max(i_depths_to_interp_)], desired_depths_[np.max(i_depths_existing_)])
        )
    if np.min(i_depths_to_interp_)<np.min(i_depths_existing_):
        print("WARNING: extrapolation occurs: extrapolating for depth=%.2f when min existing depth=%.2f"
            %(desired_depths_[np.min(i_depths_to_interp_)], desired_depths_[np.min(i_depths_existing_)])
        )
    this_shank_lfp_[i_depths_to_interp_, :] = interpolator(i_depths_to_interp_)
    return this_shank_lfp_

def plot_helper(plot_t_, plot_h_, shank_data_, xlim_seconds_, stim_onset_, figtitle_str_, figsavepath_str_):
    plt.figure()
    vmax = np.max(np.abs(shank_data_))
    # plt.pcolormesh(plot_t, plot_h, this_shank_raw_ord0, cmap="seismic", vmin=-vmax, vmax=vmax, shading="auto")
    plt.contourf(plot_t_, plot_h_, shank_data_, cmap="seismic", vmin=-vmax, vmax=vmax, levels=40)
    plt.gca().invert_yaxis()
    plt.xlim(xlim_seconds_)
    plt.axvline(stim_onset_, linestyle='-.', color='k', linewidth=2)
    plt.gca().set_aspect((xlim_seconds_[1]-xlim_seconds_[0])/np.max(plot_h_)/3)
    # plt.gca().set_aspect(1)
    plt.colorbar()
    plt.title(figtitle_str_)
    plt.savefig(figsavepath_str_)
    plt.close() # plt.show()


def csds_one_session(session_spk_dir, session_templfp_dir, session_res_dir):
    
    if not os.path.exists(session_res_dir):
        os.makedirs(session_res_dir)

    with open(os.path.join(session_spk_dir, "pre_MS.json"), "r") as f:
        session_info = json.load(f)
    raw_sfreq = session_info['SampleRate']
    # probe_is_2x16 = session_info['ELECTRODE_2X16']
    trial_duration_samples = int(session_info["SequenceTime"]*session_info["SeqPerTrial"]*raw_sfreq)
    trial_start_stamps = loadmat(os.path.join(session_spk_dir, "trials_times.mat"))['t_trial_start'].squeeze() # in samples
    n_trials = trial_start_stamps.shape[0]
    if os.path.exists(os.path.join(session_spk_dir, "trial_mask.csv")):
        trial_mask = pd.read_csv(os.path.join(session_spk_dir, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
    else:
        print("INFO: trial_mask not found!")
        trial_mask = np.ones(n_trials, dtype=bool)
    

    ######################## determine electrode shank & depth
    ## TODO can be packaged into a function that returns `spacing` and `shank_depth_lut_list`( a list of dicts, each dict indicating all electrodes in shank, sorted and grouped by depth)
    geom = pd.read_csv(os.path.join(session_spk_dir, "geom.csv"), header=None).values
    if session_info['ELECTRODE_2X16']:
        GH = 30
        GW_BETWEENSHANK = 250
        nchs_vertical = 16
    else:
        GH = 25
        GW_BETWEENSHANK = 300
        nchs_vertical = 32
    n_chs = geom.shape[0]
    func_get_shank_id = lambda intan_id: geom[intan_id, 0]//GW_BETWEENSHANK
    intan_ch_ids = np.arange(n_chs)
    # shank_dicts = OrderedDict()
    shank_depth_lut_list = []
    for shank_id, list_intan_ids in sort_and_groupby(intan_ch_ids, keyfunc=func_get_shank_id):
        # shank_dicts[shank_id] = list_intan_ids
        depths = []
        intan_tuples = []
        for depth, list_intan_j in sort_and_groupby(list_intan_ids, keyfunc=lambda x: (geom[x, 1], geom[x, 0])):
            depths.append(depth)
            intan_tuples.append(list_intan_j)
        shank_depth_lut_list.append(OrderedDict(depths=depths, intan_tuples=intan_tuples, shank_id=shank_id))
    spacing = GH
    n_shanks = len(shank_depth_lut_list)
    desired_depths = np.arange(nchs_vertical)*spacing
    desired_depth_ids = list(range(nchs_vertical))
    valid_shanks = [d["shank_id"] for d in shank_depth_lut_list]


    ############ read trial-based EPHYS
    path_wholemda = os.path.join(session_templfp_dir, "converted_data.mda")
    path_dest = os.path.join(session_templfp_dir, "ephys_trial_padded_500hz.npz")
    trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 60), pad_samples=30000, save=True, read_cache=True) # 500Hz
    # trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 15), pad_samples=30000, save=True, read_cache=True) # 2000Hz
    # trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 1), pad_samples=30000, save=False, read_cache=False) # no resampling
    
    # trials_npz = np.load(path_dest)
    trials_data = trials_npz['trials_data'] # (n_trials, n_chs, n_samples)
    up = trials_npz["resample_up"]
    down = trials_npz["resample_down"]
    ephys_newfreq = raw_sfreq*up/down

    if trials_npz["pad_is_clipped"]==False:
        padding_re = int(trials_npz["pad_samples_raw"]*up/down)
        trials_data = trials_data[:, :, padding_re:-padding_re]

    ############### REJECT BAD TRIALS
    print("#trials before ePhys noise rejection:", np.sum(trial_mask))
    trial_mask_temp = reject_trials_by_ephys(trials_data, 10000, [2,4], ephys_newfreq)
    trial_mask = (trial_mask & trial_mask_temp)
    print("#trials after ePhys noise rejection:", np.sum(trial_mask))
    trials_data = trials_data[trial_mask, :, :]
    assert trials_data.shape[1]==n_chs
    n_samples = trials_data.shape[2]
    
    # trial average
    data_trial_avg = np.mean(trials_data, axis=0) # (n_ch, n_samples)
    

#region commented_out_psdplot
    # data_trial_avg = data_trial_avg - np.mean(data_trial_avg, axis=0) # mean subtraction
    # # view spectrum
    # temp_start_sec = 0
    # data_excerpted = data_trial_avg[:, int(temp_start_sec*ephys_newfreq):]
    # f, psd_trial_avg = signal.welch(data_excerpted, fs=ephys_newfreq, axis=-1, nfft=1024)
    # for i in range(n_chs):
    #     print(i)
    #     plt.figure()
    #     plt.subplot(211)
    #     plt.plot(temp_start_sec+np.arange(data_excerpted.shape[1])/ephys_newfreq, data_excerpted[i, :])
    #     plt.subplot(212)
    #     plt.semilogy(f, psd_trial_avg[i, :])
    #     plt.xlabel('frequency [Hz]')
    #     plt.ylabel('PSD [uV**2/Hz]')
    #     plt.savefig(os.path.join(fig_dir, "psd_ch{}.png".format(i)))
    #     plt.close()
    # exit(0)
    # z-score for each channel
    # data_trial_avg = zscore(data_trial_avg, axis=-1)
#endregion

    shank_lfp_trial_avg = np.empty((n_shanks, nchs_vertical, n_samples), dtype=float)
    i_depths_to_interp_by_shank = []
    for i_dict, shank_dict in enumerate(shank_depth_lut_list):
        i_depths_to_interp = []
        i_depths_existing = []
        i_shank = shank_dict['shank_id']
        depths = shank_dict['depths']
        intan_tuples = shank_dict['intan_tuples']
        this_shank_lfp = np.empty((nchs_vertical, n_samples), dtype=float)
        # for i_depth, (depth, intan_tuple) in enumerate(zip(depths, intan_tuples))
        for i_depth, depth in enumerate(desired_depths):
            # we are assuming desired_depths is sorted with smallest element first.
            ids_list = np.where(depths==depth)[0]
            if len(ids_list)==0:
                # missing electrode at this depth. 
                # TODO  interpolate
                i_depths_to_interp.append(i_depth)
            else:
                intan_tuple = intan_tuples[ids_list[0]]
                # shank_lfp_trial_avg[i_shank, i_depth, :] = np.mean(data_trial_avg[intan_tuple, :], axis=0)
                this_shank_lfp[i_depth, :] = data_trial_avg[intan_tuple[0], :]# np.mean(data_trial_avg[intan_tuple, :], axis=0)
                i_depths_existing.append(i_depth)
        if (len(i_depths_to_interp)>0):
            interp_mode = "cubic" if len(i_depths_existing)>=4 else "linear"
            this_shank_lfp = interpolate_one_shank(this_shank_lfp, i_depths_existing, i_depths_to_interp, desired_depths, interp_mode)
        shank_lfp_trial_avg[i_shank, :, :] = this_shank_lfp
        i_depths_to_interp_by_shank.append(i_depths_to_interp)
        
    # interpolation - second time (remove outlier channels)
    interpped_depths_by_shank = []
    for i_shank, _ in enumerate(shank_depth_lut_list):
        this_shank_lfp = shank_lfp_trial_avg[i_shank, :, :]
        diff1 = np.sum(np.abs(np.diff(this_shank_lfp, axis=0)), axis=-1)
        bad_channel_ids = []
        bad_tmpmask = ( diff1 > (np.mean(diff1)+2*np.std(diff1)) )
        # scan
        if bad_tmpmask[0]:
            bad_channel_ids.append(0)
        for idx_ch in range(1, bad_tmpmask.shape[0]):
            if bad_tmpmask[idx_ch-1] and bad_tmpmask[idx_ch]:
                bad_channel_ids.append(idx_ch)
        print("DEBUG shank%d has %d bad channels" % (i_shank, len(bad_channel_ids)))
        ids_interp_net = ( set(i_depths_to_interp_by_shank[i_shank]) | set(bad_channel_ids) )
        ids_existing = set(desired_depth_ids) - set(bad_channel_ids)
        if (len(bad_channel_ids)>0):
            interp_mode = "cubic" if len(i_depths_existing)>=4 else "linear"
            this_shank_lfp = interpolate_one_shank(this_shank_lfp, list(ids_existing), bad_channel_ids, desired_depths, interp_mode)
        # i_depths_to_interp_by_shank.append(list(ids_interp))
        interpped_depths_by_shank.append([desired_depths[i_dpth] for i_dpth in ids_interp_net])
        shank_lfp_trial_avg[i_shank, :, :] = this_shank_lfp
        # plt.figure()
        # plt.subplot(211)
        # plt.plot(diff1)
        # plt.axhline(np.mean(diff1)+2*np.std(diff1))
        # plt.subplot(212)
        # vmax = np.max(shank_lfp_trial_avg[i_shank, :, :])
        # plt.imshow(shank_lfp_trial_avg[i_shank, :, :].T, cmap="seismic", vmin=-vmax, vmax=vmax)
        # plt.gca().set_aspect(1/n_samples*n_chs/4)
        # plt.show()

    ############### BAND PASS FILTER / SMOOTHING
    # band_edges = [30, 100]
    # sos = signal.butter(8, np.array(band_edges)/ephys_newfreq*2, btype="bandpass", output='sos')
    # shank_lfp_trial_avg = signal.sosfilt(sos, shank_lfp_trial_avg, axis=-1)
    # Some smoothing
    shank_lfp_trial_avg = signal.savgol_filter(shank_lfp_trial_avg, 11, 2, axis=-1)
    lowess_ = lambda arr: utils_cc.lowess(np.arange(nchs_vertical), arr, r=11)
    shank_lfp_trial_avg = np.apply_along_axis(lowess_, 1, shank_lfp_trial_avg)
    # for i_ch in range(n_chs):
    #     data_trial_avg[:, i_ch] = lowess_(data_trial_avg[i_shank:, i_ch])


    ############### COMPUTE CSD
    shank_raw_ord0 = np.empty_like(shank_lfp_trial_avg)
    shank_raw_ord1 = np.empty((shank_lfp_trial_avg.shape[0], shank_lfp_trial_avg.shape[1]-1, shank_lfp_trial_avg.shape[2]))
    shank_raw_ord2 = np.empty((shank_lfp_trial_avg.shape[0], shank_lfp_trial_avg.shape[1]-2, shank_lfp_trial_avg.shape[2]))
    shank_inv_csds_r5ch = np.empty((shank_lfp_trial_avg.shape[0], shank_lfp_trial_avg.shape[1], shank_lfp_trial_avg.shape[2]))
    shank_inv_csds_rall = np.empty((shank_lfp_trial_avg.shape[0], shank_lfp_trial_avg.shape[1], shank_lfp_trial_avg.shape[2]))
    
    for i_shank in valid_shanks:
        # this_shank_raw_ord0 = inverse_csd(shank_lfp_trial_avg[i_shank, :, :], ephys_newfreq, spacing, radius_um=spacing*5)
        this_shank_raw_ord0 = shank_lfp_trial_avg[i_shank, :, :]
        this_shank_raw_ord1 = np.diff(this_shank_raw_ord0, axis=0)
        this_shank_raw_ord2 = np.diff(this_shank_raw_ord1, axis=0)
        shank_raw_ord0[i_shank, :, :] = this_shank_raw_ord0
        shank_raw_ord1[i_shank, :, :] = this_shank_raw_ord1
        shank_raw_ord2[i_shank, :, :] = this_shank_raw_ord2
        shank_inv_csds_r5ch[i_shank, :, :] = inverse_csd(this_shank_raw_ord0, ephys_newfreq, GH, radius_um=GH*5)
        shank_inv_csds_rall[i_shank, :, :] = inverse_csd(this_shank_raw_ord0, ephys_newfreq, GH)

    ############## SPATIALLY  INTERPOLATE CSD
    # interp_spacing = 5 # um
    # interp_size    = int(np.floor(desired_depths[-1]/interp_spacing))
    # interp_depths  = np.arange(interp_size)*interp_spacing
    interp_depths = desired_depths
    # clip_seconds = [1, 4]
    # shankd_csd = shank_raw_ord0[:,:,int(clip_seconds[0]*ephys_newfreq):int(clip_seconds[1]*ephys_newfreq)]
    # interp_size = 200 #shank_raw_ord0.shape[-1]
    # interp_spacing = (desired_depths[-1]-desired_depths[0])/interp_size # um
    # interp_depths = np.arange(interp_size-2)*interp_spacing
    # interpolator_csd = interp1d(desired_depths, shank_raw_ord0, axis=1, kind="linear")
    # shank_raw_ord0 = interpolator_csd(interp_depths)
    # kernel = gkern(20, sig=1)
    # shank_raw_ord0 = convolve(shank_raw_ord0, kernel[None, :, :])


    ############## SAVE AND PLOT
    np.save(os.path.join(session_res_dir, "ephys_psth_raw0.npy"), shank_raw_ord0)
    np.save(os.path.join(session_res_dir, "ephys_psth_raw1.npy"), shank_raw_ord1)
    np.save(os.path.join(session_res_dir, "ephys_csd_deri2.npy"), shank_raw_ord2)
    np.save(os.path.join(session_res_dir, "ephys_csd_inv_5ch.npy"), shank_inv_csds_r5ch)
    np.save(os.path.join(session_res_dir, "ephys_csd_inv_all.npy"), shank_inv_csds_rall)
    shank_raw_ord0 = shank_raw_ord0/1E9 # uA/m^3 to uA/mm^3
    print(shank_raw_ord0.shape)
    
    xlim_seconds = [2.3, 2.9]
    stim_onset = session_info["StimulationStartTime"]

    for i_shank in valid_shanks:
        this_shank_raw_ord0 = shank_raw_ord0[i_shank, :, :]
        this_shank_raw_ord1 = shank_raw_ord1[i_shank, :, :]
        this_shank_raw_ord2 = shank_raw_ord2[i_shank, :, :]
        this_shank_inv_csd_r5ch = shank_inv_csds_r5ch[i_shank, :, :]
        this_shank_inv_csd_rall = shank_inv_csds_rall[i_shank, :, :]
        plot_t = np.arange(this_shank_raw_ord0.shape[1])/ephys_newfreq
        

        # 0-th order (raw ephys)
        figtitle_str0 = "RawSmoothed-These depths(um) are missing and interpolated:\n"+str(interpped_depths_by_shank[i_shank])
        figsavepath_str0 = os.path.join(session_res_dir, "shank%d_nofilt_ord0.png"%(i_shank))
        plot_helper(plot_t, interp_depths, this_shank_raw_ord0, xlim_seconds, stim_onset, figtitle_str0, figsavepath_str0)

        # 1st order (raw ephys)
        figtitle_str1 = "Bipolar-These depths(um) are missing and interpolated:\n"+str(interpped_depths_by_shank[i_shank])
        figsavepath_str1 = os.path.join(session_res_dir, "shank%d_nofilt_ord1.png"%(i_shank))
        plot_helper(plot_t, interp_depths[1:], this_shank_raw_ord1, xlim_seconds, stim_onset, figtitle_str1, figsavepath_str1)

        # 2nd order (raw ephys)
        figtitle_str2 = "Laplacian-These depths(um) are missing and interpolated:\n"+str(interpped_depths_by_shank[i_shank])
        figsavepath_str2 = os.path.join(session_res_dir, "shank%d_nofilt_ord2.png"%(i_shank))
        plot_helper(plot_t, interp_depths[1:-1], this_shank_raw_ord2, xlim_seconds, stim_onset, figtitle_str2, figsavepath_str2)

        # Inverse-method CSD (R=5x vertical spacing)
        figtitle_str3 = "CSD(R=5 x VerticalSpacing)\nThese depths(um) are missing and interpolated:\n"+str(interpped_depths_by_shank[i_shank])
        figsavepath_str3 = os.path.join(session_res_dir, "shank%d_nofilt_icsd_r5ch.png"%(i_shank))
        plot_helper(plot_t, interp_depths, this_shank_inv_csd_r5ch, xlim_seconds, stim_onset, figtitle_str3, figsavepath_str3)

        # Inverse-method CSD (R=n_ch x vertical spacing)
        figtitle_str4 = "CSD(R=N_ch x VerticalSpacing)\nThese depths(um) are missing and interpolated:\n"+str(interpped_depths_by_shank[i_shank])
        figsavepath_str4 = os.path.join(session_res_dir, "shank%d_nofilt_icsd_rall.png"%(i_shank))
        plot_helper(plot_t, interp_depths, this_shank_inv_csd_rall, xlim_seconds, stim_onset, figtitle_str4, figsavepath_str4)

    # # trials_data_bpf = trials_data
    # trials_data_bpf_hil = signal.hilbert(trials_data_bpf, axis=-1)
    # trials_data_bpf_env = np.abs(trials_data_bpf_hil)
    # trials_data_bpf_phi = np.angle(trials_data_bpf_hil)
    # if trials_npz["pad_is_clipped"]==False:
    #     padding_re = int(trials_npz["pad_samples_raw"]*up/down)
    #     trials_data = trials_data[:, :, padding_re:-padding_re]
    #     trials_data_bpf = trials_data_bpf[:, :, padding_re:-padding_re]
    #     trials_data_bpf_env = trials_data_bpf_env[:, :, padding_re:-padding_re]
    #     trials_data_bpf_phi = trials_data_bpf_phi[:, :, padding_re:-padding_re]
    # n_samples = trials_data.shape[2]
    # n_trials  = trials_data.shape[0]
    # print(n_samples , n_samples / ephys_newfreq)
    # f, t, Sxx = signal.spectrogram(trials_data_bpf, nperseg=256, noverlap=192, fs=ephys_newfreq, axis=-1)
    # Sxx = zscore(Sxx, axis=-1)
    

if __name__ == "__main__":
    import config as cfg

    # session_spk_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_bc7/2021-12-06"
    # outputdir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/mytempfolder/"
    # rel_dir = "BC7/2021-12-06"
    # result_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/bc7_2112006_raw_fs500hz_230310"
    templfp_rootdir = cfg.npz_rawdir
    for lfp_rel_dir, spk_rel_dir in zip(cfg.lfp_reldirs, cfg.spk_reldirs):
        session_spkdir = os.path.join(cfg.spk_inpdir, spk_rel_dir)
        session_templfp_dir = os.path.join(templfp_rootdir, lfp_rel_dir)
        session_res_dir = os.path.join(cfg.lfp_resdir, spk_rel_dir)
        csds_one_session(session_spkdir, session_templfp_dir, session_res_dir)
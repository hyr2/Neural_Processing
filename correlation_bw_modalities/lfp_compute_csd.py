# 1 read RHD and save as MDA for accessing arbitrary segment
#     (TODO figure out a way to do downsampling by chunk while keeping the sample alignment)
# 2 get trial stamps and read corresponding segments of ePhys
# 3 downsample, filter, Hilbert.
# 4 cross-correlation with spike

import os, sys
from copy import deepcopy
import gc
import json
import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.signal as signal
import scipy.fftpack as fftpack
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import convolve

from utils_cc import inverse_csd, sort_and_groupby
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


def gkern(l, sig=1.):
    """ https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

if __name__ == "__main__":
    session_spk_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_bc7/2021-12-06"
    outputdir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/mytempfolder/"
    rel_dir = "BC7/2021-12-06"
    fig_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/tmp_figs/bc7_csd_20230301_Req5spacing"
    path_wholemda = os.path.join(outputdir, rel_dir, "converted_data.mda")
    path_dest = os.path.join(outputdir, rel_dir, "ephys_trial_padded_2000hz.npz")
    path_temp_csd = os.path.join(outputdir, rel_dir, "ephys_csdmat.npy")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

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
    valid_shanks = [d["shank_id"] for d in shank_depth_lut_list]


    ############ read trial-based EPHYS
    # trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 60), pad_samples=30000, save=True, read_cache=True) # 500Hz
    trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 15), pad_samples=30000, save=False, read_cache=True) # 2000Hz
    # trials_npz = save_trial_ephys_resample(path_wholemda, path_dest, trial_stamps=trial_start_stamps, trial_duration=trial_duration_samples, resample_factor=(1, 1), pad_samples=30000, save=False, read_cache=False) # no resampling
    
    # trials_npz = np.load(path_dest)
    trials_data = trials_npz['trials_data'] # (n_trials, n_chs, n_samples)
    up = trials_npz["resample_up"]
    down = trials_npz["resample_down"]
    
    if trials_npz["pad_is_clipped"]==False:
        padding_re = int(trials_npz["pad_samples_raw"]*up/down)
        trials_data = trials_data[:, :, padding_re:-padding_re]

    # for i_trial in range(n_trials):
    #     if np.max(trials_data[i_trial, :, :] > 900):
    #         trial_mask[i_trial] = False
    
    ephys_newfreq = raw_sfreq*up/down
    trials_data = trials_data[trial_mask, :, :]
    assert trials_data.shape[1]==n_chs
    n_samples = trials_data.shape[2]
    
    # trial average
    data_trial_avg = np.mean(trials_data, axis=0) # (n_ch, n_samples)
    
    # z-score for each channel
    # data_trial_avg = zscore(data_trial_avg, axis=-1)

    shank_lfp_trial_avg = np.empty((n_shanks, nchs_vertical, n_samples), dtype=float)
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
            interpolator = interp1d(i_depths_existing, this_shank_lfp[i_depths_existing, :], axis=0, kind='linear', assume_sorted=True, fill_value="extrapolate")
            # print("Existing:", i_depths_existing)
            # print("To interpolate", i_depths_to_interp)
            if np.max(i_depths_to_interp)>np.max(i_depths_existing):
                print("WARNING: extrapolation occurs: extrapolating for depth=%.2f when max existing depth=%.2f"
                    %(desired_depths[np.max(i_depths_to_interp)], desired_depths[np.max(i_depths_existing)])
                )
            if np.min(i_depths_to_interp)<np.min(i_depths_existing):
                print("WARNING: extrapolation occurs: extrapolating for depth=%.2f when min existing depth=%.2f"
                    %(desired_depths[np.min(i_depths_to_interp)], desired_depths[np.min(i_depths_existing)])
                )
            this_shank_lfp[i_depths_to_interp, :] = interpolator(i_depths_to_interp)
        shank_lfp_trial_avg[i_shank, :, :] = this_shank_lfp
    

    ############### BAND PASS FILTER
    # band_edges = [30, 100]
    # sos = signal.butter(8, np.array(band_edges)/ephys_newfreq*2, btype="bandpass", output='sos')
    # shank_lfp_trial_avg = signal.sosfilt(sos, shank_lfp_trial_avg, axis=-1)

    ############### COMPUTE CSD
    shank_csd = np.empty_like(shank_lfp_trial_avg)
    for i_shank in valid_shanks:
        this_shank_csd = inverse_csd(shank_lfp_trial_avg[i_shank, :, :], ephys_newfreq, spacing, radius_um=spacing*5)
        shank_csd[i_shank, :, :] = this_shank_csd
    
    ############## SPATIALLY  INTERPOLATE CSD
    # interp_spacing = 5 # um
    # interp_size    = int(np.floor(desired_depths[-1]/interp_spacing))
    # interp_depths  = np.arange(interp_size)*interp_spacing
    clip_seconds = [1, 4]
    shankd_csd = shank_csd[:,:,int(clip_seconds[0]*ephys_newfreq):int(clip_seconds[1]*ephys_newfreq)]
    interp_size = 200 #shank_csd.shape[-1]
    interp_spacing = (desired_depths[-1]-desired_depths[0])/interp_size # um
    interp_depths = np.arange(interp_size-2)*interp_spacing
    interpolator_csd = interp1d(desired_depths, shank_csd, axis=1, kind="linear")
    shank_csd = interpolator_csd(interp_depths)
    kernel = gkern(20, sig=1)
    shank_csd = convolve(shank_csd, kernel[None, :, :])


    ############## SAVE AND PLOT
    np.save(path_temp_csd, shank_csd)
    shank_csd = shank_csd/1E9 # uA/m^3 to uA/mm^3
    print(shank_csd.shape)
    
    xlim_seconds = [2.4, 2.6]
    for i_shank in valid_shanks:
        this_shank_csd = shank_csd[i_shank, :, :]
        plot_t = np.arange(this_shank_csd.shape[1])/ephys_newfreq
        plot_h = interp_depths
        plt.figure()
        vmax = np.max(np.abs(this_shank_csd))
        plt.pcolormesh(plot_t, plot_h, this_shank_csd, cmap="seismic", vmin=-vmax, vmax=vmax, shading="auto")
        plt.gca().invert_yaxis()
        plt.xlim(xlim_seconds)
        plt.axvline(session_info["StimulationStartTime"], linestyle='-.', color='k', linewidth=2)
        plt.gca().set_aspect((xlim_seconds[1]-xlim_seconds[0])/np.max(plot_h))
        # plt.gca().set_aspect(1)
        plt.colorbar()
        plt.savefig(os.path.join(fig_dir, "shank%d_nofilt.png"%(i_shank)))
        plt.show()

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
    
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import zscore

import matplotlib.pyplot as plt

def calc_t2p(waveform, fs, target_fs=100000, n_clipout=0):
    """
    Python translation of the trough2peak metric from CellExplorer 
    Original MATLAB code: https://github.com/petersenpeter/CellExplorer/blob/master/calc_CellMetrics/calc_waveform_metrics.m
    It also gives polarity
    Assumes waveform is 1d-array of primary channel waveform.
    # TODO this could be really slow if iterating through each unit to do this.
    #     Potential improvement is get `time_waveform` timestamp for once and for all. Would require spliting this function into 2.
    """
    if np.any(np.isnan(waveform)):
        raise ValueError("There is NaN value in waveform")
    if n_clipout != 0:
        waveform = waveform[n_clipout:-n_clipout]
    nsamples_raw = waveform.shape[0]
    i_center_raw = (nsamples_raw-1)//2
    time_waveform_raw = np.arange(-i_center_raw, nsamples_raw-i_center_raw)/fs*1000 # sample time stamps in ms where 0 is the spike onset
    oversampling = int(np.ceil(target_fs/fs))
    fs_over = oversampling * fs
    oversample_interval = np.mean(np.diff(time_waveform_raw))/oversampling
    oversample_nsamples = int(np.floor((time_waveform_raw[-1] - time_waveform_raw[0])/oversample_interval))
    time_waveform = np.arange(oversample_nsamples)*oversample_interval + time_waveform_raw[0]
    # time_interpolator = interp1d(time_waveform_raw, time_waveform_raw, kind='cubic', assume_sorted=True)
    # time_waveform = time_interpolator(tw_evaluatepoints)
    # return time_waveform, fs_over
    trough_mask = np.logical_and(time_waveform>=-0.25, time_waveform<=0.25)
    trough_interval = np.where(trough_mask)[0][[0,-1]]
    wave_interpolator = interp1d(time_waveform_raw, zscore(waveform), kind="cubic", assume_sorted=True)
    wave = wave_interpolator(time_waveform) # returns zscore of interpolated waveform
    wave_polarity = np.mean(wave[trough_mask]) - np.mean(wave[~trough_mask])
    if wave_polarity>0:
        wave = -wave
    trough_idx = np.argmin(wave[trough_mask]) + trough_interval[0]
    trough2peak_idx = np.argmax(wave[trough_idx:])
    # plt.figure()
    # plt.plot(time_waveform_raw, zscore(waveform), color='blue', alpha=0.5)
    # plt.plot(time_waveform, wave, color='green', marker='x')
    # plt.axvline(time_waveform[trough_idx])
    # plt.axvline(time_waveform[trough_idx+trough2peak_idx])
    # plt.show()
    trough2peak_duration = trough2peak_idx/fs_over*1000 # in milliseconds
    return trough2peak_duration, wave_polarity
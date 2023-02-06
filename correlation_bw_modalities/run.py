import os

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

import crossmod_utils as crutils

folderpath="/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/processed_data_rh8/2022-12-03"
x = []
x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial1.mat")))
x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial2.mat")))
x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial3.mat")))
x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial4.mat")))
x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial5.mat")))
x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial6.mat")))
x.append(loadmat(os.path.join(folderpath, "imaging_signals", "IOS_singleTrial7.mat")))
trial_mask = pd.read_csv(os.path.join(folderpath, "trial_mask.csv"), header=None).to_numpy().squeeze().astype(bool)
print(trial_mask.shape)
data = np.stack([xx['ios_trials_local'][trial_mask, :, :] for xx in x], axis=0) # (nROIs, nTrials, time, n_modalities)
data = np.transpose(data, axes=[0,3,1,2]) # (nROIs, nModalities, nTrials, nTime)
print(data.shape)
# for i in range(5):
#     plt.subplot(5,1,i+1)
#     plt.plot(data[:, :, i].T)
# plt.show()

# d = data[0,:,0].squeeze()
sampling_interval = 0.18
data_r, resampling_interval = crutils.resample_0phase(data, sampling_interval, 18, 5, axis=-1)
data_r = data_r - np.mean(data_r, axis=-1)[..., None] # THIS IS VERY IMPORTANT!!!
# data_r = data_r / np.std(data_r, axis=-1)[..., None]
print(data_r.shape)

plot_range = [-2.5, 2.5]
# TODO vectorize
corr_all = []
for i in range(data.shape[2]): 
    sig_a = data_r[0, 0, i, :]
    sig_b = data_r[6, 0, i, :]
    # iterate over all trials
    corr_lags, corr_res = crutils.corr_normalized(sig_a, sig_b, sampling_interval=resampling_interval, 
        unbiased=True, normalized=True)

    # corrlag_samples_mask = (corr_lags>plot_range[0])&(corr_lags<plot_range[1])
    # plt.figure(figsize=(20, 12))
    # plt.subplot(211)
    # plt.plot(np.arange(sig_a.shape[0])*resampling_interval, sig_a, marker='.', label="signal A")
    # plt.plot(np.arange(sig_b.shape[0])*resampling_interval, sig_b, marker='.', label="signal B")
    # # plt.plot(np.arange(data.shape[2])*sampling_interval, data[0, i, :, 0], marker='.', label="signal A")
    # # plt.plot(np.arange(data.shape[2])*sampling_interval, data[6, i, :, 0], marker='.', label="signal B")
    # plt.xlabel("Time (seconds)")
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(corr_lags[corrlag_samples_mask], corr_res[corrlag_samples_mask], linewidth=0.7, marker='x', label='Cross-correlation')
    # plt.xlabel("Lag (seconds) (Postive means A is earlier)")
    # plt.legend()
    # plt.show()
    corr_all.append(corr_res)

corrlag_samples_mask = (corr_lags>plot_range[0])&(corr_lags<plot_range[1])
print(np.sum(corrlag_samples_mask))
corr_all_valid = np.array(corr_all)[:, corrlag_samples_mask]
corr_avg_valid = np.mean(corr_all_valid, axis=0)
corr_std_valid = np.std(corr_all_valid, axis=0)
corr_lags_valid = corr_lags[corrlag_samples_mask]

print(corr_avg_valid.shape, corr_lags_valid.shape)

plt.figure()
plt.plot(corr_lags_valid, corr_avg_valid, linewidth=0.7, marker='x', label='mean correlation')
plt.fill_between(corr_lags_valid, corr_avg_valid-corr_std_valid, corr_avg_valid+corr_std_valid, color='orange', alpha=0.3)
plt.show()
# l_extrap = d.shape[0]//8
# resampling_interval = sampling_interval * 5 / 18 

# d_tmp = np.concatenate([d[:l_extrap][::-1], d, d[-l_extrap:][::-1]])
# l_extrap_re = int(l_extrap*18/5) 
# dr = signal.resample_poly(d_tmp, 18, 5)[l_extrap_re:-l_extrap_re]

# print("Resampling done")
# plt.plot(resampling_interval*np.arange(dr.shape[0]), dr, linewidth=0.7, marker='x', label="resampled")
# plt.plot(sampling_interval*np.arange(d.shape[0]), d, linewidth=0.7, marker='.', label="original")
# plt.legend()
# print("Rendering")
# plt.show()

# df = fftpack.fft(d)

# target_samplecount = int(d.shape[0] * 18 / 5)
# df_padded = np.concatenate(
#     [df, np.zeros(target_samplecount-d.shape[0], dtype=df.dtype)])
# print(df_padded.dtype)
# sinc_interploated = np.abs(fftpack.ifft(df_padded))
# print(f"d.shape: {d.shape}; df.shape: {df.shape}; df_padded.shape: {df_padded.shape}; sinc_interpolated: {sinc_interploated.shape}")
# interp_interval = sampling_interval*d.shape[0]/sinc_interploated.shape[0]
# plt.plot(interp_interval*np.arange(sinc_interploated.shape[0]), sinc_interploated, linewidth=0.7, marker='x', label="sinc interp'd")
# plt.plot(sampling_interval*np.arange(d.shape[0]), d, linewidth=0.7, marker='.', label="original")
# plt.legend()
# plt.show()
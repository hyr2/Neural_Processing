import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pandas as pd
import textwrap

# For each cluster calculates adaptation index and trial index. Adaptation index is Z-score of initial response averages over 60 trials. Trial index is change in firing rate between first 15 and last 15 trials.
# Filters out clusters that have <1 Hz FR during stimulation.
# Only looks at excited clusters.


def adaptation(stim_start_time,stim_end_time,firing_rate_series):

    avg_stim_FR_trial = np.zeros(len(firing_rate_series))

    for i in range(len(firing_rate_series)):
        stim_firing_rate_array = firing_rate_series[i][int(stim_start_time/0.04) : int(stim_end_time/0.04)]
        if len(stim_firing_rate_array) == 0 or np.std(stim_firing_rate_array) == 0 or np.mean(stim_firing_rate_array) < 1:
            stim_response = 0
            trial_response = 0
            return stim_response,  trial_response
        else:
            avg_stim_FR_trial[i] = np.mean(stim_firing_rate_array)

    firing_rate_avg = np.mean(firing_rate_series, axis=0)
    stim_firing_rate_avg = firing_rate_avg[int(stim_start_time/0.04) : int(stim_end_time/0.04)]
    stim_response = (stim_firing_rate_avg[:4][np.argmax(np.absolute(stim_firing_rate_avg[:4]))] - np.mean(stim_firing_rate_avg)) / np.std(stim_firing_rate_avg)

    trial_response = (np.mean(avg_stim_FR_trial[:15]) - np.mean(avg_stim_FR_trial[-15:])) / (np.mean(avg_stim_FR_trial[:15]) + np.mean(avg_stim_FR_trial[-15:]))

    return stim_response, trial_response
# Author: Thomas Kutcher

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
        avg_stim_FR_trial[i] = np.mean(stim_firing_rate_array)

    firing_rate_avg = np.mean(firing_rate_series, axis=0)
    stim_firing_rate_avg = firing_rate_avg[int(stim_start_time/0.04) : int(stim_end_time/0.04)]
    stim_firing_rate_avg_end = firing_rate_avg[int(4.5/0.04) : int(5/0.04)]
    stim_response = (stim_firing_rate_avg[:7][np.argmax(np.absolute(stim_firing_rate_avg[:7]))] - np.mean(stim_firing_rate_avg)) / np.std(stim_firing_rate_avg)
    stim_response_end = (stim_firing_rate_avg[:7][np.argmax(np.absolute(stim_firing_rate_avg[:7]))] - np.mean(stim_firing_rate_avg_end)) / np.std(stim_firing_rate_avg_end)

    if stim_response < 0:
        stim_response = -(stim_firing_rate_avg[-7:][np.argmax(np.absolute(stim_firing_rate_avg[-7:]))] - np.mean(stim_firing_rate_avg)) / np.std(stim_firing_rate_avg)
        stim_response_end = -(stim_firing_rate_avg[-7:][np.argmax(np.absolute(stim_firing_rate_avg[-7:]))] - np.mean(stim_firing_rate_avg_end)) / np.std(stim_firing_rate_avg_end)


    trial_response = (np.mean(avg_stim_FR_trial[:10]) - np.mean(avg_stim_FR_trial[-10:])) / (np.mean(avg_stim_FR_trial[:10]) + np.mean(avg_stim_FR_trial[-10:]))

    return stim_response, stim_response_end, trial_response
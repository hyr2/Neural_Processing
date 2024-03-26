#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:10:38 2024

@author: hyr2
"""

# Code taken from   https://mgm248.github.io/ephys_data_analysis_book/4_ST_Spike_Synchrony.html
# Test code for spike synchrony: spike triggered population rate

import elephant
import quantities as pq
from neo.core import AnalogSignal
from neo import SpikeTrain
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import random
import viziphant.unitary_event_analysis as vue
from scipy.stats import norm, binom
from seaborn import color_palette

#Simulate a neuron, and a neural population
np.random.seed(2022)
fs = 1000
times = np.arange(0, 2, 1/fs)
freq = 50
mean_rate = 60
osc = np.sin(2 * np.pi * times[:] * freq)
osc_rate = mean_rate * osc
osc_rate += mean_rate #make sure rate is always positiv
sim_spike_rate = AnalogSignal(np.expand_dims(osc_rate, 1), units='Hz', sampling_rate=1000*pq.Hz)
st2 = elephant.spike_train_generation.inhomogeneous_poisson_process(rate=sim_spike_rate, refractory_period=3*pq.ms)

n_pop_neurons = 10
st_pop = np.empty((0))
for n in range(0, n_pop_neurons):
    st = elephant.spike_train_generation.inhomogeneous_poisson_process(rate=sim_spike_rate, refractory_period=3*pq.ms)
    st = np.asarray(st)
    st_pop = np.concatenate((st_pop, np.squeeze(st)))
    
st_pop = SpikeTrain(np.squeeze(st_pop), units=pq.s, t_start=0 * pq.s, t_stop=(2 * pq.s))   

#Calculate stPR (uncorrected)
spike_sr = 1000
sigma=5*pq.ms
kernel = elephant.kernels.GaussianKernel(sigma=sigma)
rate = elephant.statistics.instantaneous_rate(st_pop, sampling_period=1/spike_sr * pq.s,
                                              kernel=kernel, center_kernel=True)
rate = rate / n_pop_neurons
stPRs = []
spikes2 = np.squeeze(st2)
for spike in spikes2:
    stPRs.append(float(rate[int(round(spike * spike_sr, 4))]))
    
mean_stPR = np.mean(stPRs)
print('Mean firing rate: ' + str(np.mean(rate)))
print('Mean stPR: ' + str(mean_stPR*pq.Hz))    

spike_sr = 1000
window = 100
sigma=5*pq.ms
kernel = elephant.kernels.GaussianKernel(sigma=sigma)
rate = elephant.statistics.instantaneous_rate(st_pop, sampling_period=1/spike_sr * pq.s,
                                              kernel=kernel, center_kernel=True)
rate = rate / n_pop_neurons
rate = np.asarray(rate)
stPR_segs = []
spikes2 = np.squeeze(st2)
for spike in spikes2:
    spike = float(spike)
    if spike > window/spike_sr and times[-1] - spike > window/spike_sr: #If window entirely within times analyzed
        rate_seg = rate[int(round(spike*spike_sr))-int(window/2):int(round(spike*spike_sr))+int(window/2)]
        stPR_segs.append(rate_seg)
    
stPRs_arr = np.asarray(stPR_segs)
stPRs_arr = np.squeeze(stPRs_arr)

plt.plot(np.arange(-window/2, window/2, 1), np.mean(stPRs_arr,0))
plt.xlabel('Time lag (ms)')
plt.ylabel('Population firing rate (Hz)')
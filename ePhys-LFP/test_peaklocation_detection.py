#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:46:27 2021

@author: hyr2
"""

from scipy.io import loadmat
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt
import os
from Support import *

file_mat = '/run/media/hyr2/Data/Data/FH-BC6/2021-09-28/ePhys/whisker4/Processed/Spectrogram_mat/Chan29.mat'
df = loadmat(file_mat)
Time_MUA = df['time_MUA']
MUA_ndPSD = df['MUA_ndPSD']

Time_MUA = np.reshape(Time_MUA,(Time_MUA.size,))
MUA_ndPSD = np.reshape(MUA_ndPSD,(MUA_ndPSD.size,))

MUA_depth_peak = np.zeros((32,4,2),dtype = np.single)


MUA_depth_peak[0,0,:] = detect_peak_basic(MUA_ndPSD,2)

plt.figure()
plt.plot(Time_MUA,MUA_ndPSD)
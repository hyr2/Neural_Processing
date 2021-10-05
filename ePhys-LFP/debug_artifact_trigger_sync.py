#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:58:22 2021

@author: hyr2
"""

import os, sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
from Support import *
sys.path.append(os.path.join(os.getcwd(),'Intan-File-Reader'))
from load_intan_rhd_format import read_data

filename = '/run/media/hyr2/Data/Data/test/data_211002_153810.rhd'

result = read_data(filename)

arr_trig = result['board_dig_in_data'] 
arr_stim = result['board_adc_data']
arr_trig = np.reshape(arr_trig,(arr_trig.size,))
arr_stim = np.reshape(arr_stim,(arr_stim.size,))
t = result['t_dig']
# t = np.reshape(t,[1,t.size])

# plt.plot(t,arr_stim)
# plt.plot(t,arr_trig)



plt.plot(t, arr_stim)
arr_trig = np.multiply(arr_trig,1)
plt.plot(t, arr_trig+1.2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 00:16:12 2021

@author: hyr2
"""

# Temp script to run Spectrogram.py and Cortical_depth.py
import os, sys
sys.path.append(os.getcwd())
from Spectrogram import Spectrogram_main
from Cortical_depth import Cortical_depth_main
from Bin2Chan import Bin2Chan_main
from Bin2Trials import Bin2Trials_main

CMR_flag = 1                            # Reject common mode noise
chan_knob = 1                           # Set it to 2 for 2x16
normalize_PSD_flag = 1                  # (PSD-PSD_bl)/PSD_bl
pk_thresh = 0.7                         # Use 0.7 for awake and 25 for anesthetized
activation_window = 0.5                 # Analysis time window after activation (0.5 s)

# b1
source_dir = '/run/media/hyr2/Data_OverflowB/BC6/2021-11-2/ePhys/'
# Bin2Chan_main(source_dir,CMR_flag)
# Bin2Trials_main(source_dir)
# Spectrogram_main(source_dir, pk_thresh, normalize_PSD_flag,activation_window,chan_knob)     
# Cortical_depth_main(source_dir)

# day 2
source_dir = '/run/media/hyr2/Data_OverflowB/BC6/2021-11-11/ePhys/'
# Bin2Chan_main(source_dir,CMR_flag)
# Bin2Trials_main(source_dir)
# Spectrogram_main(source_dir, pk_thresh, normalize_PSD_flag,activation_window,chan_knob)  
# Cortical_depth_main(source_dir)

# day 9
source_dir = '/run/media/hyr2/Data_OverflowB/BC6/2021-11-18/ePhys/'
# Bin2Chan_main(source_dir,CMR_flag)
# Bin2Trials_main(source_dir)
# Spectrogram_main(source_dir, pk_thresh, normalize_PSD_flag,activation_window,chan_knob)  
# Cortical_depth_main(source_dir)

# day 14
source_dir = '/run/media/hyr2/Data_OverflowB/BC6/2021-11-23/ePhys/'
# Bin2Chan_main(source_dir,CMR_flag)
# Bin2Trials_main(source_dir)
# Spectrogram_main(source_dir, pk_thresh, normalize_PSD_flag,activation_window,chan_knob)  
# Cortical_depth_main(source_dir)


# day 21
source_dir = '/run/media/hyr2/Data_OverflowB/BC6/2021-11-30/ePhys/'
# Bin2Chan_main(source_dir,CMR_flag)
# Bin2Trials_main(source_dir)
# Spectrogram_main(source_dir, pk_thresh, normalize_PSD_flag,activation_window,chan_knob)  
# Cortical_depth_main(source_dir)

# day 35
source_dir = '/run/media/hyr2/Data_OverflowB/BC6/2021-12-14/ePhys/'
# Bin2Chan_main(source_dir,CMR_flag)
# Bin2Trials_main(source_dir)
# Spectrogram_main(source_dir, pk_thresh, normalize_PSD_flag,activation_window,chan_knob)  
# Cortical_depth_main(source_dir)

# day 49
source_dir = '/run/media/hyr2/Data_OverflowB/BC6/2021-12-30/ePhys/'
# Bin2Chan_main(source_dir,CMR_flag)
# Bin2Trials_main(source_dir)
# Spectrogram_main(source_dir, pk_thresh, normalize_PSD_flag,activation_window,chan_knob) 
# Cortical_depth_main(source_dir)

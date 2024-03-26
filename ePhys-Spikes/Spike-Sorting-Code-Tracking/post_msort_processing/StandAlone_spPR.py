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
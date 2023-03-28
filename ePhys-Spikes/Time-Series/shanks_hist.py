#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:46:00 2023

@author: hyr2-office
"""

# This script is used to plot histogram of shank distances

import numpy as np
from matplotlib import pyplot as plt

#bbc5


distances_all = np.array([250,1000,700,300,400,800,1200,180,180,400,180,500,1000,650,250,300,800,350,350,700,350,420,700,1000],dtype = np.int32)

bins_edges = np.arange(100,1200,75)
Hist = np.histogram(distances_all)
plt.hist(distances_all, bins=bins_edges)
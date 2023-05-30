#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:50:41 2023

@author: hyr2-office
"""

import numpy as np
from matplotlib import pyplot as plt
import os

filename_save = '/home/hyr2-office/Documents/Paper/Single-Figures-SVG/Fig1_updated/subfigures/'

ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus13']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.plot(x_axis*1000,y11,'r-',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-cluster13.svg'),format = 'svg')
plt.figure()
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/12-23-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus16']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.plot(x_axis*1000,y11,'r-',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-12-23-22-cluster16.svg'),format = 'svg')
plt.figure()
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus13']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.plot(x_axis*1000,y11,'r-',linewidth = 2.3)
plt.axis('off')
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/12-23-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus16']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.plot(x_axis*1000,y11,'r--',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-both_pyr_1.svg'),format = 'svg')


ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus310']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.plot(x_axis*1000,y11,'b-',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-10-17-22-cluster310.svg'),format = 'svg')
plt.figure()
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/12-23-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus322']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.plot(x_axis*1000,y11,'b-',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-12-23-22-cluster322.svg'),format = 'svg')
plt.figure()
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/10-17-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus310']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.plot(x_axis*1000,y11,'b-',linewidth = 2.3)
plt.axis('off')
ss = np.load('/home/hyr2-office/Documents/Data/NVC/RH-7/12-23-22/all_waveforms_by_cluster.npz',allow_pickle = True)
y1 = ss['clus322']
y11 = np.mean(y1,axis = 0)
x_axis = np.linspace(0,3.33e-3,num = 100)
plt.plot(x_axis*1000,y11,'b--',linewidth = 2.9)
plt.axis('off')
plt.savefig(os.path.join(filename_save,'rh7-both_PV_2.svg'),format = 'svg')
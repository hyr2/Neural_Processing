import os

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from utils.read_mda import readmda
from pycorrelate import pcorrelate

####
CCG_DURATION = 120 # ms, sum of both sides
CCG_BINSIZE  = 0.4 # ms
folderpath = r"D:\Rice-Courses\neuroeng_lab\codes\stroke_proj_postproc\data_msort\NVC\BC7\12-06-2021\ePhys\Processed\msorted"
####

ccg_nbins = int(CCG_DURATION/CCG_BINSIZE)
ccg_bins = np.linspace(-CCG_DURATION/2, CCG_DURATION/2, num=ccg_nbins+1)
SAMPLE_FREQ = loadmat(os.path.join(folderpath, "info.mat"))['sample_freq'][0][0]

# read firings.mda
firings = readmda(os.path.join(folderpath, "firings.mda"))
n_clus = int(np.max(firings[2,:]))
spike_times_all = firings[1,:]
spike_labels = firings[2,:].astype(int)
spike_times_by_clus =[[] for i in range(n_clus)]
spike_count_by_clus = np.zeros((n_clus,))
for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
    spike_times_by_clus[spk_lbl-1].append(spk_time-1)
for i in range(n_clus):
    spike_times_by_clus[i] = np.array(spike_times_by_clus[i])/SAMPLE_FREQ*1000
    spike_count_by_clus[i] = spike_times_by_clus[i].shape[0]

while True:
    tmp = input("Input a pair of cluster IDs to calculate CCG:")
    if (tmp.startswith("z")):
        break
    tmp_split = tmp.split(' ')
    if len(tmp_split)<2:
        print("USAGE: CLUS1 CLUS2")
        continue
    c1, c2 = int(tmp_split[0]), int(tmp_split[1])
    tmp_ccg = pcorrelate(spike_times_by_clus[c1], spike_times_by_clus[c2], ccg_bins, normalize=False)
    plt.figure()
    plt.bar(0.5*(ccg_bins[1:]+ccg_bins[:-1]), tmp_ccg, color='k')
    plt.axvline(0, color='red')
    plt.title("CCG %d vs. %d\n(spike counts: %d - %d)" % (c1, c2, spike_count_by_clus[c1], spike_count_by_clus[c2]))
    plt.tight_layout()
    plt.show()

print("Exited")


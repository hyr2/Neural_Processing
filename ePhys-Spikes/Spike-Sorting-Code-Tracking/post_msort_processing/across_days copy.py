from cProfile import label
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

datestrs = ["12-06-2021", "12-07-2021", "12-09-2021", "12-12-2021", "12-17-2021", "12-24-2021", "12-31-2021", "01-07-2022"]

MATFOLDERs = ["/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/NVC/BC7/%s/firing_rate_by_channels_220111_refined_clusters"%(datestr) for datestr in datestrs]

auc_daily_series_by_shank = [[] for _ in range(4)]
for datestr, matfolder in zip(datestrs, MATFOLDERs):
    for i_shank in range(4):
        matname = os.path.join(matfolder, "valid_normalized_spike_rates_by_channels_shank%d.mat"%(i_shank))
        try:
            auc = loadmat(matname)['area_under_normalized_curve_during_stim']
            if auc.shape[0]==0:
                auc_daily_series_by_shank[i_shank].append(0)
            else:
                auc_daily_series_by_shank[i_shank].append(np.mean(auc))
        except:
            auc_daily_series_by_shank[i_shank].append(0)
shankname = 'ABCD'
plt.figure()
for i in range(4):
    plt.plot(auc_daily_series_by_shank[i], label="Shank "+shankname[i])
plt.legend()
plt.xticks(np.arange(len(datestrs)), datestrs, rotation=45)

plt.title("BC7 - Absolute change in firing rate during stim")
plt.tight_layout()
plt.show()
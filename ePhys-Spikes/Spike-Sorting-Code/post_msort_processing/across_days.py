from cProfile import label
from matplotlib import gridspec
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
session_folder = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/NVC/BC7"

figsavepath = os.path.join(session_folder, "population_stat_figs")
if not os.path.exists(figsavepath):
    os.makedirs(figsavepath)

# datestrs = ["01-06-2022", "01-08-2022", "01-12-2022", "01-17-2022"]
datestrs = ["12-06-2021", "12-07-2021", "12-09-2021", "12-12-2021", "12-17-2021", "12-24-2021", "12-31-2021", "01-07-2022"]

MATFOLDERs = ["%s/%s/"%(session_folder, datestr) for datestr in datestrs]

total_nclus_by_shank = []
single_nclus_by_shank = []
multi_nclus_by_shank = []
act_nclus_by_shank = []
inh_nclus_by_shank = []
nor_nclus_by_shank = []
for datestr, matfolder in zip(datestrs, MATFOLDERs):
    matname = os.path.join(matfolder, "population_stat.mat")
    stat = loadmat(matname)
    total_nclus_by_shank.append(stat['total_nclus_by_shank'].squeeze())
    single_nclus_by_shank.append(stat['single_nclus_by_shank'].squeeze())
    multi_nclus_by_shank.append(stat['multi_nclus_by_shank'].squeeze())
    act_nclus_by_shank.append(stat['act_nclus_by_shank'].squeeze())
    inh_nclus_by_shank.append(stat['inh_nclus_by_shank'].squeeze())
    nor_nclus_by_shank.append(stat['nor_nclus_by_shank'].squeeze())

total_nclus_by_shank = np.stack(total_nclus_by_shank)
single_nclus_by_shank = np.stack(single_nclus_by_shank)
multi_nclus_by_shank = np.stack(multi_nclus_by_shank)
act_nclus_by_shank = np.stack(act_nclus_by_shank)
inh_nclus_by_shank = np.stack(inh_nclus_by_shank)
nor_nclus_by_shank = np.stack(nor_nclus_by_shank)

shankname = 'ABCD'
for i in range(4):
    ind = np.arange(len(datestrs))
    width=0.35
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(10,15)
    ax = fig.add_subplot(gs[2:8, :])
    p1 = ax.bar(ind, single_nclus_by_shank[:,i], width, label='Single Units')
    p2 = ax.bar(ind, multi_nclus_by_shank[:,i], width, 
                bottom=single_nclus_by_shank[:,i], label='Multi Units')
    # p3 = ax.bar(
    #     ind, 
    #     total_nclus_by_shank[:,i]-single_nclus_by_shank[:,i]-multi_nclus_by_shank[:,i], 
    #     width,
    #     bottom=single_nclus_by_shank[:,i]+multi_nclus_by_shank[:,i])
    ax.legend()
    # ax.bar_label(p1, label_type='center')
    # ax.bar_label(p2, label_type='center')
    # ax.bar_label(p3)

    ax.set_xticks(ind) 
    ax.set_xticklabels(datestrs, rotation=45)
    ax.set_ylabel("Cluster Counts")

    ax.set_title("Shank %s"%(shankname[i]))
    # plt.tight_layout()
    plt.savefig(os.path.join(figsavepath, "shank%s_units.png"%(shankname[i])))

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(10,15)
    ax = fig.add_subplot(gs[2:8, :])
    # p2 = ax.bar(ind, total_nclus_by_shank[:,i], width, label='Total # Clusters')
    p1 = ax.bar(ind, act_nclus_by_shank[:,i], width, label='Activated')
    p2 = ax.bar(ind, inh_nclus_by_shank[:,i], width, 
                bottom=act_nclus_by_shank[:,i], label='Inhibited')
    p2 = ax.bar(ind, nor_nclus_by_shank[:,i], width, 
                bottom=act_nclus_by_shank[:,i]+inh_nclus_by_shank[:,i], label='Non Responsive') 
    # p3 = ax.bar(
    #     ind, 
    #     total_nclus_by_shank[:,i]-single_nclus_by_shank[:,i]-multi_nclus_by_shank[:,i], 
    #     width,
    #     bottom=single_nclus_by_shank[:,i]+multi_nclus_by_shank[:,i])
    ax.legend()
    # ax.bar_label(p1, label_type='center')
    # ax.bar_label(p2, label_type='center')
    # ax.bar_label(p3)

    ax.set_xticks(ind) 
    ax.set_xticklabels(datestrs, rotation=45)
    ax.set_ylabel("Cluster Counts")

    ax.set_title("Shank %s"%(shankname[i]))
    # plt.tight_layout()
    plt.savefig(os.path.join(figsavepath, "shank%s_stim.png"%(shankname[i])))
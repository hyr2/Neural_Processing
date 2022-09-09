"""
    Post-processing on mountainsort result
    Customized for mice stroke project
    jz103 Dec 20 2021
"""
import numpy as np
# from scipy.io import loadmat
import json
import os
import matplotlib
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from time import time
from copy import deepcopy
import gc
from scipy.special import softmax

from utils.read_mda import readmda

# settings
F_SAMPLE = 20e3 # Hz
FIGDIRNAME = "figs_allclus_waveforms"
session_folder = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/NVC/BC7/12-12-2021"
# waveforms = loadmat(os.path.join(session_folder, "templates.mat"))['templates']
# waveforms1 = readmda(os.path.join(session_folder, "templates.mda"))
# firings = loadmat(os.path.join(session_folder, "firing.mat"))['firing']

matplotlib.rcParams['font.size'] = 22

def main_postprocess_and_viz(session_folder):
    """ main function for post processing and visualization"""
    #%% read clustering metrics file and perform rejection TODO improve rejection method; current version SUCKS
    MAP_PATH = os.path.join(session_folder, "geom.csv")
    figpath = os.path.join(session_folder, FIGDIRNAME)
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    with open(os.path.join(session_folder, "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)

    clus_metrics_list = x['clusters']
    n_clus = len(clus_metrics_list)
    clus_labels = 1 + np.arange(n_clus)
    clus_keep_mask = np.ones((n_clus,)).astype(bool)

    refrac_violation_score = [k['metrics']['refractory_violation_2msec'] for k in clus_metrics_list]
    refrac_violation_score = np.array(refrac_violation_score)

    firing_rates = np.array([k['metrics']['firing_rate'] for k in clus_metrics_list])
    isolation_score = np.array([k['metrics']['isolation'] for k in clus_metrics_list])
    noise_overlap_score = np.array([k['metrics']['noise_overlap'] for k in clus_metrics_list])
    peak_amplitudes = np.array([k['metrics']['peak_amplitude'] for k in clus_metrics_list])

    # keep 70% of the clusters by refractory period criterion
    id_refrac_thresh = np.argsort(refrac_violation_score)[int(0.7*n_clus)]
    clus_keep_mask[refrac_violation_score>refrac_violation_score[id_refrac_thresh]] = False 

    clus_keep_mask[firing_rates<0.01] = False

    # keep 90% of the clusters by isolation criterion
    id_isolation_thresh = np.argsort(isolation_score)[int((1-0.9)*n_clus)]
    clus_keep_mask[isolation_score<isolation_score[id_isolation_thresh]] = False 

    # keep 70% of the clusters by isolation criterion
    id_noise_overlap_thresh = np.argsort(noise_overlap_score)[int(0.7*n_clus)]
    clus_keep_mask[noise_overlap_score>noise_overlap_score[id_noise_overlap_thresh]] = False 

    # clus_labels = clus_labels[clus_keep_mask]
    n_clus_accepted = np.sum(clus_keep_mask)
    print("Accepted %d clusters out of %d" % (n_clus_accepted, n_clus))


    # %% 
    # ################### read firing stamps, template and continuous waveforms from MountainSort outputs and some processing
    firings = readmda(os.path.join(session_folder, "firings.mda")).astype(np.int64)
    waveforms = readmda(os.path.join(session_folder, "templates.mda")).astype(np.float64)
    # firings = loadmat(os.path.join(session_folder, "firings.mat"))['firings']
    # waveforms = loadmat(os.path.join(session_folder, "templates.mat"))['templates']
    n_ch = waveforms.shape[0]
    wavepeaks = np.max(np.abs(waveforms), axis=1)
    wavepeak_weights = softmax(0.1*wavepeaks, axis=0)
    signed_wavepeaks = np.zeros_like(wavepeaks)
    for i_ch in range(signed_wavepeaks.shape[0]):
        for i_clus in range(signed_wavepeaks.shape[1]):
            tmp_idx = np.argmax(np.abs(waveforms[i_ch, :, i_clus]))
            signed_wavepeaks[i_ch, i_clus] = waveforms[i_ch, tmp_idx, i_clus]
    geom = pd.read_csv(MAP_PATH, header=None).values

    peak_amplitudes_argsort = np.argsort(peak_amplitudes)
    peak_amplitude_ranks = np.zeros(n_clus)
    peak_amplitude_ranks[peak_amplitudes_argsort] = np.arange(n_clus) # rank from low to high
    peak_amplitude_ranks = peak_amplitude_ranks.astype(int)
    # print("positive spikes:", np.sum(peak_amplitudes>0))


    # get spike stamp for all clusters (in SAMPLEs not seconds)
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
    spike_times_by_clus =[[] for i in range(n_clus)]
    spike_count_by_clus = np.zeros((n_clus,))
    for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
        spike_times_by_clus[spk_lbl-1].append(spk_time)
    for i in range(n_clus):
        spike_times_by_clus[i] = np.array(spike_times_by_clus[i])
        spike_count_by_clus[i] = spike_times_by_clus[i].shape[0]

    # get primary channel for each label
    cnt_multi_prim = 0 
    pri_ch_lut = -1*np.ones(n_clus, dtype=int)
    for (spk_ch, spk_lbl) in zip(firings[0,:], spike_labels):
        if pri_ch_lut[spk_lbl-1]==-1:
            pri_ch_lut[spk_lbl-1] = spk_ch-1
        elif pri_ch_lut[spk_lbl-1] != spk_ch-1:
            print("*", end='')
            cnt_multi_prim += 1
    print(cnt_multi_prim)
    # exit(0)

    # estimate cluster locations
    clus_coordinates = np.zeros((n_clus, 2))
    for i_clus in range(n_clus):
        weights = wavepeak_weights[:,i_clus]
        clus_coordinates[i_clus, :] = np.sum(weights[:,None] * geom, axis=0)

    # get ISI histogram for each cluster
    n_bins=100
    isi_vis_max = 100
    isi_bin_edges = np.linspace(0, isi_vis_max, n_bins+1) # in millisec; 1ms per bin
    isi_hists = np.zeros((n_clus, isi_bin_edges.shape[0]-1))
    for i_clus in range(n_clus):
        isi = np.diff(spike_times_by_clus[i_clus]) / F_SAMPLE * 1000 # ISI series in millisec
        isi_hist_this, _ = np.histogram(isi, bins=isi_bin_edges)
        isi_hists[i_clus, :] =isi_hist_this

    # get spike amplitudes with time
    spk_amp_series = []
    filt_signal = readmda(os.path.join(session_folder, "filt.mda")) # heck of a big file
    for i_clus in range(n_clus):
        prim_ch = pri_ch_lut[i_clus]
        tmp_spk_stamp = spike_times_by_clus[i_clus].astype(int)
        tmp_amp_series = deepcopy(filt_signal[prim_ch, tmp_spk_stamp])
        spk_amp_series.append(tmp_amp_series)
    del(filt_signal)
    gc.collect()

    #%% viz
    ################################## VISUALIZATION
    
    # some variables for plot-config 
    final_stamp_time = firings[1, -1] / F_SAMPLE
    
    # color code cluster amplitudes
    cmap = get_cmap("viridis", n_clus)

    # scatter-size code cluster amplitudes
    smap = np.logspace(np.log10(10), np.log10(80), num=n_clus) * 20

    #### viz cluster locations
    fig1 = plt.figure(figsize=(12, 32))
    gs_ovr = gridspec.GridSpec(32,12, figure=fig1)
    ax_loc = fig1.add_subplot(gs_ovr[:, :6])
    ax_loc.scatter(geom[:,0], geom[:,1], s=24, color='blue')
    ax_loc.scatter(\
            clus_coordinates[:, 0], clus_coordinates[:, 1], \
            marker='.', c=peak_amplitude_ranks, \
            cmap=cmap, vmin=0, vmax=n_clus, \
            s=smap[peak_amplitude_ranks], alpha=.5\
            )
    ax_loc.set_aspect("equal")
    ax_loc.set_xlim(-20, 920)
    ax_loc.set_ylim(-20, 800)
    ax_loc.set_xlabel("x-coordinate (um)")
    ax_loc.set_ylabel("y-coordinate (um)")
    for i_ch in range(n_ch):
        x, y = geom[i_ch,:]
        plot_row, plot_col = (31-int(y/25)), (int(x/300))
        ax = fig1.add_subplot(gs_ovr[plot_row, 8+plot_col])
        idx_clusters = np.where(pri_ch_lut==i_ch)[0]# list(filter(lambda x: pri_ch_lut[x]==i_ch), np.arange(n_clus))
        for idx_clus in idx_clusters:
            tmp_lineconst = np.ones(spike_times_by_clus[idx_clus].shape)
            ax.scatter(spike_times_by_clus[idx_clus]/F_SAMPLE, spk_amp_series[idx_clus], \
                c=tmp_lineconst*peak_amplitude_ranks[idx_clus], cmap=cmap, vmin=0, vmax=n_clus, \
                s=1 \
                )
        ax.set_xlim(0, final_stamp_time)
        if plot_col>0:
            ax.set_yticks([])
        if plot_row<31:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (second)")
    
    plt.savefig(os.path.join(figpath, "location.png"))
    plt.close()

    #### plot spike stamps
    # fig3 = plt.figure()
    # # for clus_label in clus_labels:
    # for i_clus in range(n_clus):
    #     # i_clus = clus_label - 1
    #     # tmp_idx = np.argmax(wavepeaks[:, i_clus])
    #     tmp_lineconst = np.ones(spike_times_by_clus[i_clus].shape)
    #     plt.scatter(\
    #         spike_times_by_clus[i_clus]/F_SAMPLE, spk_amp_series[i_clus], marker='.', \
    #         c=tmp_lineconst*peak_amplitude_ranks[i_clus], vmin=0, vmax=n_clus, cmap=cmap,\
    #         #s=tmp_lineconst*smap[peak_amplitude_ranks[i_clus]], alpha=0.5 \
    #         s=0.3, alpha=0.5\
    #         )
    # plt.xlabel("Time (Seconds)")
    # plt.ylabel("uV")
    # plt.savefig(os.path.join(figpath, "stamps.png"))
    # plt.close()

    #### viz comprehensive plots (including template waveform across channels) for all clusters
    ts = time()
    fig_size_scale = 1
    for i_clus_plot in range(n_clus):
        y_scale = np.max(np.abs(waveforms[:, :, i_clus_plot]))
        fig2 = plt.figure(figsize=(40*fig_size_scale,34*fig_size_scale))
        prim_ch = pri_ch_lut[i_clus_plot] # look up primary channel
        gs_ovr = gridspec.GridSpec(32, 48, figure=fig2)
        
        # plot channel & cluster location viz
        ax_loc_viz = fig2.add_subplot(gs_ovr[:, :8])
        # all channels
        ax_loc_viz.scatter(geom[:,0], geom[:,1], s=24, color='blue')
        # highlight primary channel
        ax_loc_viz.scatter( \
            [geom[prim_ch,0]], [geom[prim_ch,1]], \
            marker='s', s=24, color='orange' \
            )
        # location of current cluster
        ax_loc_viz.scatter(\
            [clus_coordinates[i_clus_plot,0]], [clus_coordinates[i_clus_plot,1]], \
            marker="x", s=smap[int(peak_amplitude_ranks[i_clus_plot])], color='orange', \
            ) 
        # all clusters
        ax_loc_viz.scatter(\
            clus_coordinates[:, 0], clus_coordinates[:, 1], \
            marker='.', c=peak_amplitude_ranks, \
            cmap=cmap, vmin=0, vmax=n_clus, \
            s=smap[peak_amplitude_ranks], alpha=.5\
            )
        ax_loc_viz.set_aspect("equal")
        ax_loc_viz.set_xlim(-20, 920)
        ax_loc_viz.set_ylim(-20, 800)
        ax_loc_viz.set_xlabel("x-coordinate (um)")
        ax_loc_viz.set_ylabel("y-coordinate (um)")
        
        # plot waveforms
        # gs_waveforms = gridspec.GridSpecFromSubplotSpec(16, 2, subplot_spec=gs_ovr[:, 4:8]) # syntactically correct?
        for i_ch in range(n_ch):
            x, y = geom[i_ch,:]
            plot_row, plot_col = (31-int(y/25)), (int(x/300))
            ax = fig2.add_subplot(gs_ovr[plot_row, 10+plot_col*3:13+plot_col*3])# plt.subplot(16,2,plot_row*2+plot_col+1)
            ax.plot(\
                np.arange(waveforms.shape[1])/F_SAMPLE*1000, \
                waveforms[i_ch, :, i_clus_plot], \
                # label="Coordinate (%d,%d)" % (x, y),\
                # color=cmap(peak_amplitude_ranks[i_clus_plot]) \
                )
            ax.set_ylim(-1*y_scale, y_scale)
            if plot_col>0:
                ax.set_yticks([])
            if plot_row<31:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (ms)")

            # ax.legend(fontsize=13)
            # ax.set_title("Coordinate (%d,%d)" % (x, y), fontsize=10)
        
        # plot ISI histogram
        ax_isihist = fig2.add_subplot(gs_ovr[:6, 24:])
        ax_isihist.bar(0.5+np.arange(n_bins), isi_hists[i_clus_plot,:], width=1.)
        ax_isihist.set_xticks(isi_bin_edges[::10])
        ax_isihist.set_xticklabels(isi_bin_edges[::10])
        ax_isihist.set_xlabel("ISI (ms)")
        ax_isihist.set_ylabel("Count")
        ax_isihist.set_xlim(0, isi_vis_max)
        ax_isihist.set_title("ISI histogram")
        
        # plot Amplitude series
        ax_ampstamp = fig2.add_subplot(gs_ovr[7:13, 24:])
        ax_ampstamp.scatter(spike_times_by_clus[i_clus_plot]/F_SAMPLE, spk_amp_series[i_clus_plot], s=1)
        ax_ampstamp.set_xlim(0, final_stamp_time)
        ax_ampstamp.set_xlabel("Time (sec)")
        ax_ampstamp.set_ylabel("Transient amplitude (uV)")
        
        # print annotations
        ax_text = fig2.add_subplot(gs_ovr[15:, 24:])
        str_annot  = "Cluster label: %d\n" % (clus_labels[i_clus_plot])
        str_annot += "Average firing rate: %.2f (Total spike count: %d)\n" % (firing_rates[i_clus_plot], spike_count_by_clus[i_clus_plot])
        str_annot += "Isolation score: %.4f\n" % (isolation_score[i_clus_plot])
        str_annot += "Noise overlap score: %.4f\n" % (noise_overlap_score[i_clus_plot])
        str_annot += "Refractory 2ms violation score: %.4f\n" % (refrac_violation_score[i_clus_plot])
        str_annot += "Auto-accepted=%d\n" % (clus_keep_mask[i_clus_plot])
        ax_text.text(0.5, 0.5, str_annot, va="center", ha="center", fontsize=22)
        # plt.suptitle("Cluster %d, kept=%d" % (i_clus_plot+1, clus_keep_mask[i_clus_plot]), fontsize=25)
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(os.path.join(figpath, "waveform_clus%d.png"%(clus_labels[i_clus_plot])))
        plt.close()
        print(i_clus_plot+1)
    print("Plotting done in %f seconds" % (time()-ts))


if __name__ == '__main__':
    main_postprocess_and_viz(session_folder)
    




# %%


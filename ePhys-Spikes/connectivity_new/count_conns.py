import os, sys
from time import time
from copy import deepcopy
import json
import re
from datetime import datetime
from itertools import groupby
from functools import reduce
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from scipy.io import loadmat

sys.path.append("../Spike-Sorting-Code/preprocess_rhd")
from utils.read_mda import readmda
sys.path.append("../Spike-Sorting-Code/post_msort_processing/utils")
from waveform_metrics import calc_t2p
import config as cfg



def read_postproc_data(session_spk_dir: str, mdaclean_temp_dir: str, session_connec_dir: bool) -> dict:
    """ read post processed data
    read_connec : whether to read connectivity info
    Returns dict:
    ret['spike_count']
    ret['peak_amplitudes']
    ret['accpet_mask'] : single units
    ret['locations'] : unit locations
    ret['prim_chs']
    Optional : ret['connectivity'] (n_connecs, 3): where the first 2 columns are unit indices already mapped back to MS output
    Process one entire session.
    Assumes the 'label' field in combine_metrics_new.json is 1..n_clus_uncurated
    """

    ### read clustering metrics file
    with open(os.path.join(session_spk_dir, "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)
    ret = {}
    with open(os.path.join(session_spk_dir, "pre_MS.json"), "r") as f:
        session_info = json.load(f)
    # get shank# for each channel
    geom = pd.read_csv(os.path.join(session_spk_dir, "geom.csv"), header=None).values
    if session_info['ELECTRODE_2X16']:
        GW_BETWEENSHANK = 250
    else:
        GW_BETWEENSHANK = 300
    ch2shank = pd.read_csv(os.path.join(mdaclean_temp_dir, "ch2shank.csv"), header=None).values
    template_waveforms = readmda(os.path.join(mdaclean_temp_dir, "templates_clean.mda")).astype(np.int64)
    map_clean2original = pd.read_csv(os.path.join(mdaclean_temp_dir, "map_clean2original_labels.csv"), header=None).values
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus)
    spikewidths = [calc_t2p(template_waveforms[pri_ch_lut[i_], :, i_], session_info["SampleRate"]) for i_ in range(pri_ch_lut.shape[0])]
    spikewidths = np.array([k[0] for k in spikewidths])
    # read putative connectivity
    connecs = pd.read_csv(os.path.join(session_connec_dir, "connecs.csv"), header=None).values
    connecs[:, :2] = connecs[:,:2]-1
    ret["connecs"] = connecs
    n_ex_cons_from = np.zeros(pri_ch_lut.shape[0])
    n_in_cons_from = np.zeros(pri_ch_lut.shape[0])
    for src, snk, contype in connecs[:, :]:
        if contype > 0:
            n_ex_cons_from[src] += 1
        else:
            n_in_cons_from[src] += 1
        src_original_id = map_clean2original[src]
        snk_original_id = map_clean2original[snk]
        src_shanknum    = ch2shank[pri_ch_lut[src]]
        snk_shanknum    = ch2shank[pri_ch_lut[snk]]
    ret["spikewidths"] = spikewidths
    ret["n_in_cons_from"] = n_in_cons_from
    ret["n_ex_cons_from"] = n_ex_cons_from
    print(n_in_cons_from)
    print(n_ex_cons_from)
    return ret


def get_datetime(x):
    tmp = re.match(r"([0-9]+-[0-9]+-[0-9]+)", x)
    if tmp is None:
        return None
    try:
        date = datetime.strptime(tmp[1], "%y-%m-%d")
    except Exception as e:
        date = datetime.strptime(tmp[1], "%Y-%m-%d")
    return date

def get_datetime_str(x):
    # TODO optimize: get_datetime is called repetitively
    datetime_obj = get_datetime(x)
    if datetime_obj is None:
        return None
    return datetime.strftime(datetime_obj, "%Y-%m-%d")

def plot_src_conns_vs_spkwidth(data_dicts, figpath):
    fig, axes = plt.subplots(2, 1)
    for data_dict in data_dicts:
        axes[0].plot(data_dict["spikewidths"], data_dict["n_ex_cons_from"], ".", markersize=1, color="red")
        axes[1].plot(data_dict["spikewidths"], data_dict["n_in_cons_from"], ".", markersize=1, color="blue")
    axes[1].set_xlabel("Spike width (ms)")
    axes[0].set_ylabel("number of post-synaptic targets")
    os.makedirs(figpath, exist_ok=True)
    plt.savefig(os.path.join(figpath, "src_conns_vs_spkwidth.png"))
    plt.close()

def boxplot_src_conns_vs_spkwidth(data_dicts, figpath):
    
    all_spikewidths = np.concatenate([dd["spikewidths"] for dd in data_dicts])
    all_n_ex_cons_from = np.concatenate([dd["n_ex_cons_from"] for dd in data_dicts])
    all_n_in_cons_from = np.concatenate([dd["n_in_cons_from"] for dd in data_dicts])

    has_connec_mask = np.logical_or(all_n_ex_cons_from>0, all_n_in_cons_from>0)
    all_spikewidths = all_spikewidths[has_connec_mask]
    all_n_ex_cons_from = all_n_ex_cons_from[has_connec_mask]
    all_n_in_cons_from = all_n_in_cons_from[has_connec_mask]

    inh_mask = (all_spikewidths<0.47)
    groups = [
        "Excitations\nfrom I cell",
        "Inhibitions\nfrom I cell",
        "Excitations\nfrom E cell",
        "Inhibitions\nfrom E cell",
        ]
    datasets = [
        all_n_ex_cons_from[inh_mask],
        all_n_in_cons_from[inh_mask],
        all_n_ex_cons_from[~inh_mask],
        all_n_in_cons_from[~inh_mask],
    ]
    data_means = [np.mean(dataset) for dataset in datasets]
    data_sdems = [np.std(dataset)/np.sqrt(dataset.shape[0]) for dataset in datasets]
    fig, ax = plt.subplots()
    # ax.boxplot(datasets, positions=[0,1,3,4])
    ax.bar([0,1,3,4], data_means, width=0.4, yerr=data_sdems, 
        color="white",edgecolor='black'
    )
    ax.set_xticks([0,1,3,4])
    ax.set_xticklabels(groups)
    ax.set_ylabel("Synapse counts")
    plt.savefig(os.path.join(figpath, "src_conns_vs_spkwidth_box.png"))
    plt.close()


def plot_src_conns_vs_spkwidth_hist2d(data_dicts, figpath):
    spikewidths_ex_src = []
    spikewidths_in_src = []
    n_ex_cons_from = []
    n_in_cons_from = []
    for data_dict in data_dicts:
        spkws = data_dict["spikewidths"]
        nexfr = data_dict["n_ex_cons_from"]
        ninfr = data_dict["n_in_cons_from"]
        emask = (nexfr>0)
        imask = (ninfr>0)
        spikewidths_ex_src.extend(spkws[emask].tolist())
        spikewidths_in_src.extend(spkws[imask].tolist())
        n_ex_cons_from.extend(nexfr[emask].tolist())
        n_in_cons_from.extend(nexfr[imask].tolist())
    n_con_max = max(max(n_ex_cons_from), max(n_in_cons_from))
    spkwidths_max = max(max(spikewidths_ex_src), max(spikewidths_in_src))
    spkwidths_min = min(min(spikewidths_ex_src), min(spikewidths_in_src))
    round_custom = lambda x: np.round(x/0.05)*0.05
    bin_edges_spkw = np.arange(round_custom(spkwidths_min), round_custom(spkwidths_max)+0.01, 0.05)
    bin_edges_con = np.arange(0.5, n_con_max+1.0, 1.0)
    hist2d_ex, _, _ = np.histogram2d(spikewidths_ex_src, n_ex_cons_from, bins=(bin_edges_spkw, bin_edges_con))
    hist2d_in, _, _ = np.histogram2d(spikewidths_in_src, n_in_cons_from, bins=(bin_edges_spkw, bin_edges_con))
    # normalize for each spikewidth range
    hist2d_ex = hist2d_ex / np.sum(hist2d_ex, axis=1)[:, None]
    hist2d_in = hist2d_in / np.sum(hist2d_in, axis=1)[:, None]
    # plot
    fig, axes = plt.subplots(2, 1)
    me = axes[0].pcolormesh(bin_edges_spkw, bin_edges_con, hist2d_ex.T)
    plt.colorbar(me, ax=axes[0])
    mi = axes[1].pcolormesh(bin_edges_spkw, bin_edges_con, hist2d_in.T)
    plt.colorbar(mi, ax=axes[1])
    axes[1].set_xlabel("Spike width (ms)")
    axes[0].set_ylabel("Percentage of excitatory\npost-synaptic targets")
    axes[1].set_ylabel("Percentage of inhibitory\npost-synaptic targets")
    os.makedirs(figpath, exist_ok=True)
    plt.savefig(os.path.join(figpath, "src_conns_vs_spkwidth_hist2d.png"))
    plt.close()


if __name__ == "__main__":
    data_dicts = []
    for reldir in cfg.spk_reldirs:
        data_folder = os.path.join(cfg.spk_inpdir, reldir)
        temp_folder = os.path.join(cfg.mda_tempdir, reldir)
        data_dicts.append(read_postproc_data(data_folder, temp_folder, temp_folder))
    boxplot_src_conns_vs_spkwidth(data_dicts, cfg.mda_tempdir)


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
import multiprocessing

import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat

sys.path.append("../Spike-Sorting-Code/preprocess_rhd")
from utils.read_mda import readmda
import config as cfg

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.size']=30

# SHANK_SPACING = 300

N_THREADS = 7


# CELLTYPE_PYRAMIDAL = 0
# CELLTYPE_NARROW_INTER = 1
# CELLTYPE_WIDE_INTER = 2
# CELLTYPES = {
#     CELLTYPE_PYRAMIDAL: "Pyramidal",
#     CELLTYPE_NARROW_INTER: "Narrow Inter",
#     CELLTYPE_WIDE_INTER: "Wide Inter"
# }

# SHANK_SWAP_BC = False

# def get_segment_index(segment_name: str) -> int:
#     return int(re.search("seg([0-9]+)", segment_name)[1])

def read_postproc_data(session_spk_dir: str, mdaclean_temp_dir, session_connec_dir: bool) -> dict:
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

    geom = pd.read_csv(os.path.join(session_spk_dir, "geom.csv"), header=None).values
    ret["geom"] = geom
    if session_info['ELECTRODE_2X16']:
        GW_BETWEENSHANK = 250
        ret["SHANK_LAYOUT"] = "2X16"
    else:
        GW_BETWEENSHANK = 300
        ret["SHANK_LAYOUT"] = "1X32"
    # get shank# for each channel
    ch2shank = pd.read_csv(os.path.join(mdaclean_temp_dir, "ch2shank.csv"), header=None).values
    map_clean2original_labels = pd.read_csv(os.path.join(mdaclean_temp_dir, "map_clean2original_labels.csv"), header=None).values
    template_waveforms = readmda(os.path.join(mdaclean_temp_dir, "templates_clean.mda")).astype(np.int64)
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus) this is apparently 0-based
    ret["pri_ch_lut"] = pri_ch_lut
    # read putative connectivity
    connecs = pd.read_csv(os.path.join(session_connec_dir, "connecs.csv"), header=None).values
    connecs[:, :2] = connecs[:,:2]-1 # force 0-based indexing (because that result connec.csv file is from MATLAB)
    ret["connecs"] = connecs # 0-based indexing
    n_c_wishank = 0
    n_c_bwshank = 0
    for src, snk in connecs[:, :2]:
        if (ch2shank[pri_ch_lut[src]]) == (ch2shank[pri_ch_lut[snk]]):
            n_c_wishank += 1
        else:
            n_c_bwshank += 1
    ret["n_con_bw_shank"] = n_c_bwshank
    ret["n_con_wi_shank"] = n_c_wishank
    ret["ch2shank"] = ch2shank
    ret["map_clean2original_labels"] = map_clean2original_labels
    mono_res_mat = loadmat(os.path.join(session_connec_dir, "mono_res.cellinfo.mat"))
    ccg_binsize_ms = mono_res_mat["mono_res"][0][0]['binSize'][0][0]*1000
    ccg_nbins = mono_res_mat["mono_res"][0][0]['ccgR'].shape[0]
    ret["ccg_bincenters_ms"] = (np.arange(ccg_nbins)-(ccg_nbins//2))*ccg_binsize_ms
    ret["ccg"] = mono_res_mat["mono_res"][0][0]['ccgR']
    ret["template_waveforms"] = template_waveforms
    ret["abs_amplitudes"] = np.max(template_peaks_single_sided, axis=0) # (n_clus,)
    ret["sample_freq"] = session_info["SampleRate"]

    return ret

def helper_plotwaveforms(i_unit_, chs_, ax_array_, template_waveforms_, geom_, gh_, gw_bwshank_, fs_, color_, max_amp_):
    waveform_len = template_waveforms_.shape[1]
    for iter_ch, ch_id in enumerate(chs_):
        x, y = geom_[ch_id, :]
        row = int(y)//gh_
        col = int((int(x)%gw_bwshank_) > 0) # should be either 0 or 1 for 2x16 and constantly 0 for 1x32
        ax_array_[row][col].plot(
            np.arange(waveform_len)/fs_*1000,
            template_waveforms_[ch_id, :, i_unit_],
            color=color_
        )
        ax_array_[row][col].set_ylim([-max_amp_, max_amp_])
        ax_array_[row][col].set_yticks([])
        ax_array_[row][col].set_xticks([])

def plot_one_connec_(pp_dict_, figsavedir, i_connec):
    """
        Given relevant data of a session (pp_dict_) and a putative connection pair:
            * If they are in the same shank: plot their waveform distributions on that shank
            * If they are in different shanks: plot their waveform distributions on primary shank
            * Plot CCG and ACGs
            * Annotate edge type; source and sink.
    """
    import matplotlib
    matplotlib.font_manager._get_font.cache_clear()
    ccg_bincenters_ms = pp_dict_["ccg_bincenters_ms"]
    ch2shank = pp_dict_["ch2shank"]
    map_clean2orig_labels = pp_dict_["map_clean2original_labels"]
    connecs = pp_dict_["connecs"]
    pri_ch_lut = pp_dict_["pri_ch_lut"]
    geom = pp_dict_["geom"]
    n_chs = geom.shape[0]
    template_waveforms = pp_dict_["template_waveforms"]
    template_abs_amps  = pp_dict_["abs_amplitudes"]
    # waveform_len = template_waveforms.shape[1]
    f_sample = pp_dict_["sample_freq"]
    ccg = pp_dict_["ccg"]
    ccg_binwidth = ccg_bincenters_ms[1]-ccg_bincenters_ms[0]
    print("   ", figsavedir, i_connec)
    src_id_cur = connecs[i_connec, 0]
    snk_id_cur = connecs[i_connec, 1]
    con_type   = ("EXCITATORY" if connecs[i_connec, 2]==1 else "INHIBITORY")
    src_shank  = ch2shank[pri_ch_lut[src_id_cur]]
    snk_shank  = ch2shank[pri_ch_lut[snk_id_cur]]
    src_chs = list(filter(lambda c: ch2shank[c]==src_shank, list(range(n_chs))))
    snk_chs = list(filter(lambda c: ch2shank[c]==snk_shank, list(range(n_chs))))
    max_amp = max(template_abs_amps[src_id_cur], template_abs_amps[snk_id_cur])

    fig = plt.figure(figsize=(24, 32))
    gs_ovr = GridSpec(nrows=32, ncols=12)

    if pp_dict_["SHANK_LAYOUT"] == "2X16":
        waveform_axes = [[fig.add_subplot(gs_ovr[r, c]) for c in range(2)] for r in range(16)] # (16, 2) python List
        for axss in waveform_axes:
            for ax in axss:
                ax.set_xticks([])
                ax.set_yticks([])
        gw_bwshank = 250
        gh = 30
    elif pp_dict_["SHANK_LAYOUT"] == "1X32":
        waveform_axes = [[fig.add_subplot(gs_ovr[r, c]) for c in range(1)] for r in range(32)] # (32, 1) python List
        for axss in waveform_axes:
            for ax in axss:
                ax.set_xticks([])
                ax.set_yticks([])
        gw_bwshank = 300
        gh = 25
    else:
        raise NotImplementedError("Bad Shank Layout")

    # plot neuron waveform distribution
    helper_plotwaveforms(src_id_cur, src_chs, waveform_axes, template_waveforms, geom, gh, gw_bwshank, f_sample, "k", max_amp)
    helper_plotwaveforms(snk_id_cur, snk_chs, waveform_axes, template_waveforms, geom, gh, gw_bwshank, f_sample, ("red" if con_type=="EXCITATORY" else "blue"), max_amp)
    ax_text  = fig.add_subplot(gs_ovr[:6, 4:])
    textstr  = "Edge type: %14s\n" % (con_type)
    textstr += "Source unit ID[Curated]=%3d Label[MS]=%3d Shank=%d(black)  \n" % (src_id_cur, map_clean2orig_labels[src_id_cur], src_shank)
    textstr += "Sink   unit ID[Curated]=%3d Label[MS]=%3d Shank=%d(colored)\n"% (snk_id_cur, map_clean2orig_labels[snk_id_cur], snk_shank)
    ax_text.text(0.5, 0.5, textstr, va="center", ha="center", fontsize=28)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    ax_ccg   = fig.add_subplot(gs_ovr[6:12, 4:])
    ax_ccg.bar(ccg_bincenters_ms, ccg[:, src_id_cur, snk_id_cur], width=ccg_binwidth, color="k")
    ax_ccg.set_xticks([])
    ax_ccg_src = fig.add_subplot(gs_ovr[12:18, 4:])
    ax_ccg_src.bar(ccg_bincenters_ms, ccg[:, src_id_cur, src_id_cur], width=ccg_binwidth, color="k")
    ax_ccg_src.set_xticks([])
    ax_ccg_snk = fig.add_subplot(gs_ovr[18:24, 4:])
    ax_ccg_snk.bar(ccg_bincenters_ms, ccg[:, snk_id_cur, snk_id_cur], width=ccg_binwidth, color=("red" if con_type=="EXCITATORY" else "blue"))
    plt.savefig(os.path.join(figsavedir, "connec%3d.png")%(i_connec))
    plt.close()



# def get_datetime(x):
#     tmp = re.match(r"([0-9]+-[0-9]+-[0-9]+)", x)
#     if tmp is None:
#         return None
#     try:
#         date = datetime.strptime(tmp[1], "%y-%m-%d")
#     except Exception as e:
#         date = datetime.strptime(tmp[1], "%Y-%m-%d")
#     return date

# def get_datetime_str(x):
#     # TODO optimize: get_datetime is called repetitively
#     datetime_obj = get_datetime(x)
#     if datetime_obj is None:
#         return None
#     return datetime.strftime(datetime_obj, "%Y-%m-%d")


def proc_main(spk_dir_root, mda_tempdir_root, connec_dir_root, spk_reldirs, mda_reldirs):
    # animal_dir = mda_reldirs[0].split("/")[0]
    # pp_dicts = []
    parallel_args = []
    for mda_reldir, spk_reldir in zip(mda_reldirs, spk_reldirs):
        session_spk_dir = os.path.join(spk_dir_root, spk_reldir)
        session_mda_dir = os.path.join(mda_tempdir_root, mda_reldir)
        session_con_dir = os.path.join(connec_dir_root, mda_reldir)
        pp_dict = read_postproc_data(session_spk_dir, session_mda_dir, session_con_dir)
        # plot_each_connec_(pp_dict, session_con_dir)
        for i_c in range(pp_dict["connecs"].shape[0]):
            parallel_args.append((pp_dict, session_con_dir, i_c))

    with multiprocessing.Pool(N_THREADS) as pool:
        pool.starmap(plot_one_connec_, parallel_args)

import config as cfg
proc_main(
    cfg.spk_inpdir,
    cfg.mda_tempdir,
    cfg.con_resdir,
    cfg.spk_reldirs,
    cfg.mda_reldirs
)

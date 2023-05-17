import os, sys
# from time import time
# from copy import deepcopy
import json
# import re
# from datetime import datetime
# from itertools import groupby
# from functools import reduce
# import warnings
# from collections import OrderedDict
import multiprocessing
# import typing
import argparse

import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
import numba

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

# TODO explore possibility of explicitly specifying JITted types
# TODO `perheps templates` will not be used
@numba.jit(nopython=True, parallel=True, fastmath=True)
def _get_curate_mask(
    spk_counts: np.ndarray, acg_contrast: np.ndarray,
    ccg_counts: np.ndarray, ccg_contrast: np.ndarray, conns: np.ndarray,
    param_spk_count_th: int, param_acg_contrast_th: float, 
    param_ccg_count_th: int, param_ccg_contrast_th: float,
    ):
    curation_mask = np.empty(conns.shape[0], dtype=np.bool_)
    n_units = spk_counts.shape[0]
    exc_out_degrees = np.zeros((n_units,), dtype=np.int64)
    inh_out_degrees = np.zeros((n_units,), dtype=np.int64)
    conn_mat        = np.zeros((n_units, n_units), dtype=np.int64)
    for i_conn in range(conns.shape[0]):
        src = conns[i_conn][0]
        snk = conns[i_conn][1]
        conn_type = conns[i_conn][2]

        reject_flag = (
            # reject connections with few ACG total spike counts in either unit
            # (acg_counts[src]<param_acg_count_th) or # the reference neuron can be sparse
            (spk_counts[snk]<param_spk_count_th) or
            # reject connections with few CCG total spike counts
            (ccg_counts[src, snk]<param_ccg_count_th) or
            # reject connections with large ACG bin max in either unit
            (acg_contrast[src]>=param_acg_contrast_th) or
            (acg_contrast[snk]>=param_acg_contrast_th) or
            # reject connections with large CCG bin max
            (ccg_contrast[src, snk]>=param_ccg_contrast_th)
        )
        # same directed pair cannot be both excitatory and inhibitory
        # if so then this pair is rejected.
        reject_flag |= ( (conn_mat[src][snk] != 0) and 
                        (conn_mat[src][snk] != conn_type) 
                    )
        curation_mask[i_conn] = not(reject_flag)
        if reject_flag:
            continue
        # if the connection is not rejected, then count the type:
        conn_mat[src][snk] = conn_type # record connection type
        if conn_type>0:
            exc_out_degrees[src] += 1
        else:
            inh_out_degrees[src] += 1
    
    return curation_mask, exc_out_degrees, inh_out_degrees


def curate_one_session(
    conns: np.ndarray, ccgs: np.ndarray,
    spike_counts: np.ndarray, templates: np.ndarray,
    param_acg_count_th: int, param_ccg_count_th: int,
    param_acg_contrast_th: int, param_ccg_contrast_th: int):
    """
    Curate the connections detected by CellExplorer for one session.
    Parameters
    --------
    conns : (n_connections, 3) Each row indicates a connection described 
        by 3-tuple (src, snk, type[+1: exc; -1: inh])
    ccgs : (n_units, n_units, n_ccg_bins) pair wise CCG tensor
    templates: (n_channels, len_waveform, n_units) templates of the units
    """
    n_units = templates.shape[2]
    assert ccgs.shape[0]==ccgs.shape[1] and ccgs.shape[0]==n_units
    
    diag_ids_tmp = np.arange(n_units, dtype=np.int64)
    acgs = ccgs[diag_ids_tmp, diag_ids_tmp, :] # (n_units, len_ccg)
    # ccg_nbins = ccgs.shape[2]
    acg_counts = np.sum(acgs, axis=1) # (n_units,)
    acg_binmed = np.clip(np.median(acgs, axis=1), a_min=1, a_max=None)
    acg_binhom = np.max(acgs, axis=1) / acg_binmed# (n_units,)
    # part_ks = [ccg_nbins-3, ccg_nbins-2, ccg_nbins-1]
    # ccgs_sorted = np.partition(ccgs, kth=part_ks, axis=2)
    # acgs_sorted = ccgs_sorted[diag_ids_tmp, diag_ids_tmp, :]
    # acg_binmax  = np.sum(acgs_sorted[:, :-3])
    
    ccg_counts = np.sum(ccgs, axis=2) # (n_units, n_units)
    ccg_binmed = np.clip(np.median(ccgs, axis=2), a_min=1, a_max=None)
    ccg_binhom = np.max(ccgs, axis=2) / ccg_binmed # (n_units, n_units)
    
    curation_mask, exc_out_degrees, inh_out_degrees = _get_curate_mask(
        spike_counts, acg_binhom, ccg_counts, ccg_binhom, conns,
        param_acg_count_th, param_acg_contrast_th, param_ccg_count_th,
        param_ccg_contrast_th
    )
    
    return curation_mask, acg_counts, acg_binhom, ccg_counts, ccg_binhom, \
        exc_out_degrees, inh_out_degrees



def process_postproc_data(session_spk_dir: str, mdaclean_temp_dir: str, session_connec_dir: str) -> dict:
    """ process post processed data
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
    spike_counts = np.array([c["metrics"]["num_events"] for c in x["clusters"]])
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
    map_clean2original_labels = pd.read_csv(os.path.join(mdaclean_temp_dir, "map_clean2original_labels.csv"), header=None).values.squeeze()
    template_waveforms = readmda(os.path.join(mdaclean_temp_dir, "templates_clean.mda")).astype(np.int64)
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus) this is apparently 0-based
    ret["pri_ch_lut"] = pri_ch_lut
    spike_counts = spike_counts[map_clean2original_labels-1].astype(np.int64)
    ret["spike_counts"] = spike_counts
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
    ret["ccg"] = mono_res_mat["mono_res"][0][0]['ccgR'].transpose([1,2,0]).copy(order="C") # force contiguous (n_units, n_units, ccg_nbins) memory layout
    ret["template_waveforms"] = template_waveforms
    ret["abs_amplitudes"] = np.max(template_peaks_single_sided, axis=0) # (n_clus,)
    ret["sample_freq"] = session_info["SampleRate"]
    # conn_curate_mask, acg_counts, acg_binmax, ccg_counts, ccg_binmax = \
    curate_outputs = curate_one_session(
        connecs, ret["ccg"], ret["spike_counts"], template_waveforms,
        cfg.curate_params["PARAM_SPK_COUNT_TH"],
        cfg.curate_params["PARAM_CCG_COUNT_TH"],
        cfg.curate_params["PARAM_ACG_CONTRAST_TH"],
        cfg.curate_params["PARAM_CCG_CONTRAST_TH"]
    )
    ret["conn_keep_mask"] = curate_outputs[0] # (n_connecs,)
    ret["acg_counts"]     = curate_outputs[1] # (n_units,)
    ret["acg_contrast"]   = curate_outputs[2] # (n_units,)
    ret["ccg_counts"]     = curate_outputs[3] # (n_units, n_units)
    ret["ccg_contrast"]   = curate_outputs[4] # (n_units, n_units)
    ret["exc_outdeg"]     = curate_outputs[5] # (n_units,)
    ret["inh_outdeg"]     = curate_outputs[6] # (n_units,)

    pd.DataFrame(ret["conn_keep_mask"].astype(int)).to_csv(
        os.path.join(session_connec_dir, "conn_keep_mask.csv"),
        index=False, header=False
    )
    print("# connections Kept/total=%d/%d"%(np.sum(ret["conn_keep_mask"]), ret["conn_keep_mask"].shape[0]))
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
    ccg_bincenters_ms     = pp_dict_["ccg_bincenters_ms"]
    ch2shank              = pp_dict_["ch2shank"]
    map_clean2orig_labels = pp_dict_["map_clean2original_labels"]
    connecs               = pp_dict_["connecs"]
    pri_ch_lut            = pp_dict_["pri_ch_lut"]
    geom                  = pp_dict_["geom"]
    n_chs                 = geom.shape[0]
    template_waveforms    = pp_dict_["template_waveforms"]
    template_abs_amps     = pp_dict_["abs_amplitudes"]
    f_sample              = pp_dict_["sample_freq"]
    spk_counts            = pp_dict_["spike_counts"]
    ccg                   = pp_dict_["ccg"]
    acg_counts            = pp_dict_["acg_counts"]
    acg_contrast          = pp_dict_["acg_contrast"]
    ccg_counts            = pp_dict_["ccg_counts"]
    ccg_contrast          = pp_dict_["ccg_contrast"]
    conn_keep_mask        = pp_dict_["conn_keep_mask"]
    exc_outdegs           = pp_dict_["exc_outdeg"] 
    inh_outdegs           = pp_dict_["inh_outdeg"] 
    # all relevant fields in `pp_dict_` are extracted by now.
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
    textstr  = "Edge type: %14s; Accepted=%1d\n" % (con_type, conn_keep_mask[i_connec])
    textstr += "Source unit ID[Curated]=%3d Label[MS]=%3d Shank=%2d(black)  \n" % (src_id_cur, map_clean2orig_labels[src_id_cur], src_shank)
    textstr += "                         TotalSpikes=%4d, ACG Contrast=%3.1f\n" % (spk_counts[src_id_cur], acg_contrast[src_id_cur])
    textstr += "                    Exc. OutDegrees=%4d, Inh. OutDegrees=%4d\n" % (exc_outdegs[src_id_cur], inh_outdegs[src_id_cur])
    textstr += "Sink   unit ID[Curated]=%3d Label[MS]=%3d Shank=%2d(colored)\n" % (snk_id_cur, map_clean2orig_labels[snk_id_cur], snk_shank)
    textstr += "                         TotalSpikes=%4d, ACG Contrast=%3.1f\n" % (spk_counts[snk_id_cur], acg_contrast[snk_id_cur])
    textstr += "Connection:          CCG TotalSpikes=%4d, CCG Contrast=%3.1f\n" % (ccg_counts[src_id_cur, snk_id_cur], ccg_contrast[src_id_cur, snk_id_cur])
    ax_text.text(0.5, 0.5, textstr, va="center", ha="center", fontsize=28)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    ax_ccg   = fig.add_subplot(gs_ovr[6:12, 4:])
    ax_ccg.bar(ccg_bincenters_ms, ccg[src_id_cur, snk_id_cur, :], width=ccg_binwidth, color="k")
    ax_ccg.axvline(0, linewidth=0.7, color='g', linestyle='-.')
    ax_ccg.set_xticks([])
    ax_ccg_src = fig.add_subplot(gs_ovr[12:18, 4:])
    ax_ccg_src.bar(ccg_bincenters_ms, ccg[src_id_cur, src_id_cur, :], width=ccg_binwidth, color="k")
    ax_ccg_src.axvline(0, linewidth=0.7, color='g', linestyle='-.')
    ax_ccg_src.set_xticks([])
    ax_ccg_snk = fig.add_subplot(gs_ovr[18:24, 4:])
    ax_ccg_snk.bar(ccg_bincenters_ms, ccg[snk_id_cur, snk_id_cur, :], width=ccg_binwidth, color=("red" if con_type=="EXCITATORY" else "blue"))
    ax_ccg_snk.axvline(0, linewidth=0.7, color='g', linestyle='-.')
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


def proc_main(spk_dir_root, mda_tempdir_root, connec_dir_root, spk_reldirs, mda_reldirs, do_plot):
    # animal_dir = mda_reldirs[0].split("/")[0]
    # pp_dicts = []
    parallel_args = []
    for mda_reldir, spk_reldir in zip(mda_reldirs, spk_reldirs):
        session_spk_dir = os.path.join(spk_dir_root, spk_reldir)
        session_mda_dir = os.path.join(mda_tempdir_root, mda_reldir)
        session_con_dir = os.path.join(connec_dir_root, mda_reldir)
        pp_dict = process_postproc_data(session_spk_dir, session_mda_dir, session_con_dir)
        # plot_each_connec_(pp_dict, session_con_dir)
        for i_c in range(pp_dict["connecs"].shape[0]):
            parallel_args.append((pp_dict, session_con_dir, i_c))

    if do_plot:
        with multiprocessing.Pool(N_THREADS) as pool:
            pool.starmap(plot_one_connec_, parallel_args)


parser = argparse.ArgumentParser()
parser.add_argument('--noplot', action="store_true", default=False)
args = parser.parse_args()
proc_main(
    cfg.spk_inpdir,
    cfg.mda_tempdir,
    cfg.con_resdir,
    cfg.spk_reldirs,
    cfg.mda_reldirs,
    not(args.noplot)
)

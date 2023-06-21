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
import argparse
from math import comb

import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.stats as stats


import utils_graph
sys.path.insert(0, "../Spike-Sorting-Code/preprocess_rhd")
from utils.read_mda import readmda
sys.path.insert(0, "../Spike-Sorting-Code/post_msort_processing/utils")
from waveform_metrics import calc_t2p
import config as CFG



def read_postproc_data(session_spk_dir: str, mdaclean_temp_dir: str, session_connec_dir: str, shank_def: dict, apply_curation: bool, count_type: int) -> dict:
    """ read post processed data
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
    
    # ch2shank is the map of each channel to corresponding shank. Each entry is in {0,1,2,3}.
    ch2shank = pd.read_csv(os.path.join(mdaclean_temp_dir, "ch2shank.csv"), header=None).values.squeeze()
    template_waveforms = readmda(os.path.join(mdaclean_temp_dir, "templates_clean.mda")).astype(np.int64)
    map_clean2original = pd.read_csv(os.path.join(mdaclean_temp_dir, "map_clean2original_labels.csv"), header=None).values.squeeze()-1
    unit_locations = pd.read_csv(os.path.join(session_spk_dir, "clus_locations.csv"), header=None).values.squeeze()
    unit_locations = unit_locations[map_clean2original, :]
    unit_depths    = unit_locations[:, 1]
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus)
    n_units = pri_ch_lut.shape[0]
    # spikewidths = [calc_t2p(template_waveforms[pri_ch_lut[i_], :, i_], session_info["SampleRate"]) for i_ in range(pri_ch_lut.shape[0])]
    # spikewidths = np.array([k[0] for k in spikewidths])
    # read putative connectivity
    connecs = pd.read_csv(os.path.join(session_connec_dir, "connecs.csv"), header=None).values.astype(int)
    if apply_curation:
        curate_mask = pd.read_csv(os.path.join(session_connec_dir, "conn_keep_mask.csv"), header=None).values.squeeze()
        curate_mask = (curate_mask>0).astype(bool)
        connecs = connecs[curate_mask, :]
    connecs[:, :2] = connecs[:,:2]-1
    
    # in case all connections are rejected:
    if connecs.shape[0]==0:
        connecs = np.zeros((0,3), dtype=int)
        ret["conn_exc_ratio"] = None
        ret["conn_inh_ratio"] = None
    else:
        ret["conn_exc_ratio"] = np.sum(connecs[:, 2]==1) / connecs.shape[0]
        ret["conn_inh_ratio"] = 1 - ret["conn_exc_ratio"]
    ret["connecs"] = connecs
    # ret["conn_cnt_mat"] = np.zeros((4,4), dtype=int)
    ret["conn_cnt_types"] = list(CFG.ShankLoc) # NOTE this assumes the ShankLoc enum type has values from 0 and increments by 1.
    ret["templates_clean"] = template_waveforms
    unit_shanks = ch2shank[pri_ch_lut[np.arange(n_units)]]
    unit_regions = np.array([shank_def[sk_] for sk_ in unit_shanks])

    # if not(apply_curation):
    #     print("  ====%s" % (session_connec_dir))

    conn_vertdists = np.abs(unit_depths[connecs[:, 0]] - unit_depths[connecs[:,1]])
    conn_exc_vertdists = conn_vertdists[connecs[:, 2]==1]
    conn_inh_vertdists = conn_vertdists[connecs[:, 2]==-1]
    ret["conn_exc_vertdists"] = conn_exc_vertdists
    ret["conn_inh_vertdists"] = conn_inh_vertdists

    # for i_connec, (src, snk, contype) in enumerate(connecs[:, :]):
    #     if (count_type!=0 and contype!=count_type):
    #         continue
    #     # The commented-out code counts the connections cross-region and normalize.
    #     # src_original_id = map_clean2original[src]
    #     # snk_original_id = map_clean2original[snk]
    #     # src_shanknum    = ch2shank[pri_ch_lut[src]]
    #     # snk_shanknum    = ch2shank[pri_ch_lut[snk]]
    #     # src_region      = shank_def[src_shanknum] # IntEnum instance
    #     # snk_region      = shank_def[snk_shanknum] # IntEnum instance
    #     # assert src_region==unit_regions[src] and snk_region==unit_regions[snk]
    #     # if src_region == snk_region:
    #     #     # for connections whose source and target are in the same region, we only consider them if they are in the same shank
    #     #     if src_shanknum == snk_shanknum:
    #     #         ret["conn_cnt_mat"][src_region][snk_region] += 1
    #     # else:
    #     #     ret["conn_cnt_mat"][src_region][snk_region] += 1
    #     # #print cross-shank connection
    #     if not(apply_curation) and src_shanknum != snk_shanknum:
    #         print("    ====i_connec:%3d, Source: %d, Sink: %d" % (i_connec, src, snk))
    # noramlize by number of edges maximum possible
    # mat_max_edges = np.ones((4,4), dtype=int)
    # for i_r_, src_region_ in enumerate(ret["conn_cnt_types"]):
    #     n_i_ = np.sum(unit_regions==src_region_)
    #     for j_r_, snk_region_ in enumerate(ret["conn_cnt_types"]):
    #         if i_r_==j_r_:
    #             # remember we only consider within-shank for within-region connections
    #             temp = 0
    #             shanks_oi_ = list(filter(lambda x: shank_def[x]==src_region_, range(4)))
    #             for sh_ in shanks_oi_:
    #                 n_units_in_shank_ = np.sum(unit_shanks==sh_)
    #                 temp += n_units_in_shank_ * (n_units_in_shank_-1)
    #             # mat_max_edges[i_r_, j_r_] = n_i_ * (n_i_ - 1)
    #             mat_max_edges[i_r_, j_r_] = temp
    #         else:
    #             n_j_ = np.sum(unit_regions==snk_region_)
    #             mat_max_edges[i_r_, j_r_] = n_i_ * n_j_
    # mat_max_edges = np.clip(mat_max_edges, 1, None)
    # ret["conn_density_mat"] = ret["conn_cnt_mat"].astype(float) / mat_max_edges.astype(float)
    # ret["spikewidths"] = spikewidths
    print("Edge table size:", connecs.shape)
    g = utils_graph.create_graph_no_orphans(connecs)
    n_edges = g.number_of_edges()
    
    # try:
    #     ret["graph_richclub"] = nx.rich_club_coefficient(g.to_undirected())
    # except:
    #     ret["graph_richclub"] = {0: 0}
    
    try:
        # a_ = nx.adjacency_matrix(g).todense()
        # print(a_)
        hu, au = utils_graph.calc_hits(g)
        edge_be_cent = nx.edge_betweenness_centrality(g, normalized=True, k=None)
        # print("hu:", hu)
        # print("au:", au)
        ret["graph_hits_h_array"] = hu
        ret["graph_hits_a_array"] = au
        ret["graph_edge_be_cen_array"] = np.array(list(edge_be_cent.values()))
    except Exception as e:
        print("Warning: encountered exception in centrality calculation:")
        print(e)
        ret["graph_hits_h_array"] = np.array([])
        ret["graph_hits_a_array"] = np.array([])
        ret["graph_edge_be_cen_array"] = np.array([])
    
    # Efficiency metric
    try:
        ret["graph_global_efficiency"] = nx.global_efficiency(g.to_undirected())
    except:
        ret["graph_global_efficiency"] = None
    try:
        ret["graph_local_efficiency"] = nx.local_efficiency(g.to_undirected())
    except:
        ret["graph_local_efficiency"] = None
    
    matched_microcircuits_dict = utils_graph.match_microcircuits(g, utils_graph.MICROCIRCUIT_NXGRAPHS)
    ret["matched_mc_count_dict"] = dict((k, len(v)) for k, v in matched_microcircuits_dict.items())
    mc_dens_dict = dict()
    for k, v in matched_microcircuits_dict.items():
        mc_size = utils_graph.MICROCIRCUIT_NXGRAPHS[k].number_of_edges()
        if n_edges < mc_size:
            print("n_edges=%d<mc_size=%d"%(n_edges, mc_size))
            dens_ = 0
        else:
            dens_ = len(v) / comb(n_edges, mc_size)
        mc_dens_dict[k] = dens_
    ret["matched_mc_density_dict"] = mc_dens_dict
    return ret


def get_connec_data_single_animal(animal_id, session_reldirs, session_ids, cfg_module, apply_curation, count_type):
    # data_dicts = []
    data_dict = OrderedDict()
    for i_session, session_reldir in enumerate(session_reldirs):
        print(session_reldir)
        session_spk_dir_   = os.path.join(cfg_module.spk_inpdir, session_reldir)
        mdaclean_temp_dir_ = os.path.join(cfg_module.mda_tempdir, session_reldir)
        monosyn_conn_dir_  = os.path.join(cfg_module.con_resdir, session_reldir)
        data_dict_t = read_postproc_data(session_spk_dir_, mdaclean_temp_dir_, monosyn_conn_dir_, cfg_module.shank_defs[animal_id], apply_curation, count_type)
        # conn_mat_sym = data_dict["conn_cnt_mat"] + data_dict["conn_cnt_mat"].T - np.diag(np.diag(data_dict["conn_cnt_mat"]))
        # conn_mat_sym = (data_dict["conn_density_mat"] + data_dict["conn_density_mat"].T) / 2#  - np.diag(np.diag(data_dict["conn_cnt_mat"]))
        data_dict[session_ids[i_session]] = data_dict_t
    return data_dict

def get_connec_data_all_animal(animal_session_id_dict, animal_session_reldir_dict, cfg_module, apply_curation, count_type):
    animal_data_dict = OrderedDict()
    for animal_id in animal_session_id_dict.keys():
        session_ids = animal_session_id_dict[animal_id]
        session_reldirs = animal_session_reldir_dict[animal_id]
        animal_data_dict[animal_id] = get_connec_data_single_animal(animal_id, session_reldirs, session_ids, cfg_module, apply_curation, count_type)
    return animal_data_dict




def get_datetime(x: str):
    tmp = re.match(r"([0-9]+-[0-9]+-[0-9]+)", x)
    if tmp is None:
        return None
    try:
        date = datetime.strptime(tmp[1], "%y-%m-%d") # yy-mm-dd
    except Exception as e:
        date = datetime.strptime(tmp[1], "%Y-%m-%d") # yyyy-mm-dd
    return date

def get_datetime_str(x: str):
    # TODO optimize: get_datetime is called repetitively
    datetime_obj = get_datetime(x)
    if datetime_obj is None:
        return None
    return datetime.strftime(datetime_obj, "%Y-%m-%d")

def get_delta_days(x:str, y:str):
    if isinstance(x, str) and isinstance(y, str):
        xx = get_datetime(x)
        yy = get_datetime(y)
        return int(float((yy-xx).total_seconds())/(24*3600))
    elif isinstance(x, datetime) and isinstance(y, datetime):
        return int(float((y-x).total_seconds())/(24*3600))
    else:
        raise ValueError("Bad arguments for get_delta_days()")

def round_to_nearest(lst_src, lst_ref):
    lst_ret = []
    for elem in lst_src:
        tmp = [abs(elem-eref) for eref in lst_ref]
        k = np.argmin(tmp)
        lst_ret.append(lst_ref[k])
    return lst_ret

def get_valid_days_all_animal(animal_session_id_dict, animal_dayzero_dict, days_standard):
    """
        Get valid days for all animals; assume all days/sessions in input args are sorted.
        animal_session_id_dict : a dict with animal ID as key and list of session IDs as value
        animal_dayzero_dict : a dict with animal ID as key and stroke date STRING as value
        days_standard       : a list of standard days of measurements specified in paradigm
    """
    animal_days_dict = OrderedDict()
    for animal_name, session_list in animal_session_id_dict.items():
        # print(session_list)
        animal_datezero = get_datetime(animal_dayzero_dict[animal_name])
        exp_datetimes = list(map(get_datetime, session_list))
        # print(animal_datezero)
        # print(exp_datetimes)
        exp_days_raw  = list(map(lambda x: get_delta_days(animal_datezero, x), exp_datetimes))
        # print(exp_days_raw)
        exp_days_round = round_to_nearest(exp_days_raw, days_standard)
        print(exp_days_round)
        # exp_days_uniq = [x for i, x in enumerate(exp_days_round) if exp_days_round.index(x)==i]
        exp_days_uniq = []
        exp_sess_uniq = []
        i_ = 0
        while i_ < len(exp_days_round):
            day_round = exp_days_round[i_]
            # search if there are multiple sessions round to the same day
            # (assume sorted)
            j_ = i_ + 1
            while j_ < len(exp_days_round) and exp_days_round[j_]==day_round:
                j_ +=  1
            if j_ > i_ + 1:
                # case 0: there are multiple sessions rounded to the same day
                tmp_idx = i_ + np.argmin([abs(kk - day_round) for kk in exp_days_raw[i_:j_]])
                exp_sess_uniq.append(session_list[tmp_idx])
                exp_days_uniq.append(day_round)
            else:
                # case 1: this session is rounded to a unique day
                exp_sess_uniq.append(session_list[i_])
                exp_days_uniq.append(day_round)
            i_ = j_  # increment i_
        animal_days_dict[animal_name] = OrderedDict(days=exp_days_uniq, session_ids=exp_sess_uniq)
        # print(animal_days_dict)
    return animal_days_dict



def plot_conns_for_each_animal(animal_days_dict, animal_data_dict, days_standard, fig_folder):
    """
    animal_days_dict :
        dict["animal_id"]["days"] is the list of valid days for specified animal
        dict["animal_id"]["session_ids"] is the list of session ids for corresponding days for specified animal
    animal_data_dict : dict["animal_id"]["session_id"][src_region][snk_region] is the number of conns; the matrix is symmetrical if undirected.
    """
    n_animals = len(animal_data_dict)
    os.makedirs(fig_folder, exist_ok=True)

    # richclub distribution distribution
    # fig, axes = plt.subplots(n_animals, len(days_standard), figsize=(20,12), dpi=400)
    # # axes_flat = axes.flatten()
    # # regions = list(CFG.ShankLoc)[:3]
    # for i_animal_id, animal_id in enumerate(animal_days_dict.keys()):
    #     session_ids = animal_days_dict[animal_id]["session_ids"]
    #     days        = animal_days_dict[animal_id]["days"]
    #     for i_day_id, (day, session_id) in enumerate(zip(days, session_ids)):
    #         ax_ypos = days_standard.index(day)
    #         ax = axes[i_animal_id][ax_ypos]
    #         ddict = animal_data_dict[animal_id][session_id]
    #         scores = np.array(list(ddict["graph_richclub"].values()))
    #         degrees = np.array(list(ddict["graph_richclub"].keys()))
    #         ax.plot(degrees, scores, marker=".", color='k')
    #         ax.set_title("%s - day %d" % (animal_id, day))
    # plt.tight_layout()
    # plt.suptitle("Chronic RichClub distribution for each animal")
    # os.makedirs(fig_folder, exist_ok=True)
    # plt.savefig(os.path.join(fig_folder, "Richclub.png"))
    # plt.close()


    # hits distribution
    # fig, axes = plt.subplots(n_animals, len(days_standard), figsize=(20,12), dpi=400)
    # # regions = list(CFG.ShankLoc)[:3]
    # for i_animal_id, animal_id in enumerate(animal_days_dict.keys()):
    #     session_ids = animal_days_dict[animal_id]["session_ids"]
    #     days        = animal_days_dict[animal_id]["days"]
    #     for i_day_id, (day, session_id) in enumerate(zip(days, session_ids)):
    #         ax_ypos = days_standard.index(day)
    #         ax = axes[i_animal_id][ax_ypos]
    #         ddict = animal_data_dict[animal_id][session_id]
    #         h_ = ddict["graph_hits_h_array"]
    #         a_ = ddict["graph_hits_a_array"]
    #         bin_edges = np.arange(0, 1, 11)
    #         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #         h_hist, bin_edges = np.histogram(h_)#, bin_edges)
    #         bin_centers_h = (bin_edges[:-1] + bin_edges[1:]) / 2
    #         a_hist, bin_edges = np.histogram(a_)#, bin_edges)
    #         bin_centers_a = (bin_edges[:-1] + bin_edges[1:]) / 2
    #         ax.bar(bin_centers_h, h_hist, color="orange", width=bin_centers_h[1]-bin_centers_h[0])
    #         ax.bar(bin_centers_a, -a_hist, color="blue", width=bin_centers_a[1]-bin_centers_a[0])
    #         ax.set_title("%s - day %d" % (animal_id, day))
    # plt.tight_layout()
    # plt.suptitle("Chronic HITS distribution for each animal\nOrange: hubs; Blue: authorities")
    # os.makedirs(fig_folder, exist_ok=True)
    # plt.savefig(os.path.join(fig_folder, "HITS.png"))
    # plt.close()

    
    arbitrary_single_animal = list(animal_data_dict.values())[0]
    arbitrary_single_session = list(arbitrary_single_animal.values())[0]
    matched_mc_names = list(arbitrary_single_session["matched_mc_density_dict"].keys())
    print(matched_mc_names)
    fig, axes = plt.subplots(2,3, figsize=(16,8))
    axes_flat = axes.flatten()
    for i, mcname in enumerate(matched_mc_names):
        ax = axes_flat[i]
        for i_animal_id, animal_id in enumerate(animal_days_dict.keys()):
            days = animal_days_dict[animal_id]["days"]
            session_ids = animal_days_dict[animal_id]["session_ids"]
            dataset = [animal_data_dict[animal_id][session_id]["matched_mc_density_dict"][mcname] for session_id in session_ids]
            ax.plot(days, dataset, marker='x', markersize=2.0, linewidth=1.0, label=animal_id)

        ax.set_xticks(days_standard)
        ax.legend()
        ax.set_xlabel("Day")
        ax.set_ylabel("Density")
        ax.set_title(mcname)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(fig_folder, "microcircuits.png"))
    plt.close()

# util for bucketing
def get_scalar_dataset_bucketed(datasets_bucketed_, func_getitem, filter_out_nones=True):
    """
    datasets_bucketed_ : dict of the following structure:
        datasets_bucketed_[chronic_phase] is a list of dicts; each dict is 
        keyed by name of the statitics and valued by the corresponding value
        the value could be of arbitrary type including nested dict
    func_getitem : a function that extracts the value of proper type from 
        the aforementioned dict element (usually expects a scalar)
    Returns a dict:
        scalar_dataset[chronic_phase] is a list of scalar values or other 
        well-formatted values

    """
    ret_dataset = OrderedDict()
    for chronic_phase, d_bucket in datasets_bucketed_.items():
        stat_set_aslist = [func_getitem(ddict) for ddict in d_bucket]
        if filter_out_nones:
            ret_dataset[chronic_phase] = list(filter(lambda x: x is not None, stat_set_aslist))
        else:
            ret_dataset[chronic_phase] = stat_set_aslist
    return ret_dataset

# util for statistical test
def test_buckets(datasets_bucketed, export_txt_folder, stat_name, printout=True):
    bucket_names = list(datasets_bucketed.keys())
    n_buckets = len(bucket_names)
    res_str = "%s\n"%(stat_name)
    for i in range(n_buckets-1):
        k0 = bucket_names[i]
        for j in range(i+1, n_buckets):
            k1 = bucket_names[j]
            s, p = stats.ranksums(datasets_bucketed[k0], datasets_bucketed[k1])
            # res_str += "%20s vs. %20s: statistic=%.8f, pvalue=%.4f\n" % (k0, k1, s, p)
            res_str += f"{k0} vs. {k1}: statistic={s}, pvalue={p}\n"
    export_path = os.path.join(export_txt_folder, stat_name+".txt")
    with open(export_path, "w") as f:
        f.write(res_str)
    if printout:
        print(res_str)

def boxplot_conns_for_each_animal(animal_days_dict, animal_data_dict, days_standard, fig_folder):
    """
    animal_days_dict :
        dict["animal_id"]["days"] is the list of valid days for specified animal
        dict["animal_id"]["session_ids"] is the list of session ids for corresponding days for specified animal
    animal_data_dict : dict["animal_id"]["session_id"][src_region][snk_region] is the number of conns; the matrix is symmetrical if undirected.
    """
    n_animals = len(animal_data_dict)
    os.makedirs(fig_folder, exist_ok=True)

    # TODO modularize: implement each plotting as a seperate function to reduce
    #  the length of this function
    

    buckets = OrderedDict(
        # bucketing 0
        # baseline=[-2, -1],
        # poststroke=[2, 7, 14],
        # chronic=[28, 42]
        # bucketing 1
        baseline=[-2, -1],
        subacute=[2],
        recovery=[7,14],
        chronic=[28,42]
    )
    bucket_names = list(buckets.keys())
    n_buckets = len(buckets)
    # get bucketed data
    datasets_bucketed = OrderedDict()
    for chronic_phase, bucket_days in buckets.items():
        datasets_this_phase = []
        for i_animal_id, animal_id in enumerate(animal_days_dict.keys()):
            days = np.array(animal_days_dict[animal_id]["days"])
            session_ids = np.array(animal_days_dict[animal_id]["session_ids"])
            indices_in_bucket = np.where(
                np.in1d(days, bucket_days, assume_unique=True, invert=False)
            )[0]
            session_ids_in_bucket = session_ids[indices_in_bucket]
            datasets = [animal_data_dict[animal_id][session_id] for session_id in session_ids_in_bucket]
            datasets_this_phase.extend(datasets)
            # ax.plot(days, dataset, marker='x', markersize=2.0, linewidth=1.0, label=animal_id)
        datasets_bucketed[chronic_phase] = datasets_this_phase

    # plot counts of exhibitions and inhibitions, bucketed
    exc_ratios_bucketed = get_scalar_dataset_bucketed(
        datasets_bucketed,
        lambda x: x["conn_exc_ratio"]
    ) # exc_ratios_bucketed is a list of lists; each level-2 list is a set of numbers
    fig, ax = plt.subplots()
    positions = list(range(n_buckets))
    data_means = [np.mean(dataset_) for dataset_ in exc_ratios_bucketed.values()]
    data_sdems = [np.std(dataset_)/np.sqrt(len(dataset_)) for dataset_ in exc_ratios_bucketed.values()]
    ax.bar(positions, data_means, yerr=data_sdems, color='white', edgecolor='black')
    ax.set_xticks(positions)
    ax.set_xticklabels(list(exc_ratios_bucketed.keys()))
    ax.set_ylabel("Ratio of excitatory connectivities")
    plt.savefig(os.path.join(fig_folder, "exc_ratios.png"))
    plt.close()
    

    # plot local efficiency, bucketed
    loc_effi_bucketed = get_scalar_dataset_bucketed(
        datasets_bucketed,
        lambda x: x["graph_local_efficiency"]
    ) # loc_effi_bucketed is a list of lists; each level-2 list is a set of numbers
    fig, ax = plt.subplots()
    positions = list(range(n_buckets))
    data_means = [np.mean(dataset_) for dataset_ in loc_effi_bucketed.values()]
    data_sdems = [np.std(dataset_)/np.sqrt(len(dataset_)) for dataset_ in loc_effi_bucketed.values()]
    ax.bar(positions, data_means, yerr=data_sdems, color='white', edgecolor='black')
    ax.set_xticks(positions)
    ax.set_xticklabels(list(loc_effi_bucketed.keys()))
    ax.set_ylabel("Average Local Efficiency")
    plt.savefig(os.path.join(fig_folder, "local_efficiencies.png"))
    plt.close()
    # statistical test and export results
    test_buckets(loc_effi_bucketed, fig_folder, "graph_local_efficiency")

    # plot global efficiencies, bucketed
    glo_effi_bucketed = get_scalar_dataset_bucketed(
        datasets_bucketed,
        lambda x: x["graph_global_efficiency"]
    ) # glo_effi_bucketed is a list of lists; each level-2 list is a set of numbers
    fig, ax = plt.subplots()
    positions = list(range(n_buckets))
    data_means = [np.mean(dataset_) for dataset_ in glo_effi_bucketed.values()]
    data_sdems = [np.std(dataset_)/np.sqrt(len(dataset_)) for dataset_ in glo_effi_bucketed.values()]
    ax.bar(positions, data_means, yerr=data_sdems, color='white', edgecolor='black')
    ax.set_xticks(positions)
    ax.set_xticklabels(list(glo_effi_bucketed.keys()))
    ax.set_ylabel("Global efficiency")
    plt.savefig(os.path.join(fig_folder, "global_efficiencies.png"))
    plt.close()
    # statistical test and export results
    test_buckets(glo_effi_bucketed, fig_folder, "graph_global_efficiency")

    # plot HITS distributions, bucketed
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers = bin_centers[1:] # not counting the zero bins
    bin_width = bin_edges[1] - bin_edges[0]
    def hist_helper(x: dict, k: str):
        if len(x[k])==0:
            # the case when a session does not have any connections
            return None # np.zeros(len(bin_edges)-2) 
        else:
            # not counting the bin corresponding to 0 value
            return np.histogram(x[k], bin_edges, density=True)[0][1:]*bin_width
    hits_h_bucketed = get_scalar_dataset_bucketed(
        datasets_bucketed, 
        lambda x: hist_helper(x, "graph_hits_h_array")
    )
    hits_a_bucketed = get_scalar_dataset_bucketed(
        datasets_bucketed, 
        lambda x: hist_helper(x, "graph_hits_a_array")
    )
    fig, axes = plt.subplots(n_buckets,1,figsize=(6,12))
    axes_flat = axes.ravel()
    for i_plot, (chronic_phase, datasets) in enumerate(hits_h_bucketed.items()):
        # datasets is a list of ndarrays; each ndarray is a histogram
        ax = axes_flat[i_plot]
        # filter out the sessions without any monosyn connectivity
        # datasets = list(filter(lambda x: x is not None, datasets))
        datasets_stacked = np.stack(datasets)
        mean_hist = np.mean(datasets_stacked, axis=0)
        print("  Chronic phase: %s - ratio of Hub>=0.1: %.3f"%(chronic_phase, np.sum(mean_hist)))
        sdem_hist = np.std(datasets_stacked, axis=0) / np.sqrt(len(datasets))
        # print(mean_hist, sdem_hist)
        ax.bar(
            bin_centers, mean_hist, 
            yerr=sdem_hist, color="white",edgecolor='black', width=bin_width
        )
        ax.set_title(chronic_phase)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(fig_folder, "hits_h_bucket.png"))
    plt.close()
    test_buckets(hits_h_bucketed, fig_folder, "hits_h_histogram")
    fig, axes = plt.subplots(n_buckets,1,figsize=(6,12))
    axes_flat = axes.ravel()
    for i_plot, (chronic_phase, datasets) in enumerate(hits_a_bucketed.items()):
        # datasets is a list of ndarrays; each ndarray is a histogram
        ax = axes_flat[i_plot]
        # filter out the sessions without any monosyn connectivity
        # datasets = list(filter(lambda x: x is not None, datasets))
        datasets_stacked = np.stack(datasets)
        mean_hist = np.mean(datasets_stacked, axis=0)
        print("  Chronic phase: %s - ratio of Auth>=0.1: %.3f"%(chronic_phase, np.sum(mean_hist)))
        sdem_hist = np.std(datasets_stacked, axis=0) / np.sqrt(len(datasets))
        ax.bar(
            bin_centers, mean_hist, 
            yerr=sdem_hist, color="white",edgecolor='black', width=bin_width
        )
        ax.set_title(chronic_phase)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(fig_folder, "hits_a_bucket.png"))
    plt.close()
    test_buckets(hits_a_bucketed, fig_folder, "hits_a_histogram")

    # "graph_edge_be_cen_array"
    bin_edges = np.linspace(0, 0.5, 21)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    def hist_helper(x: dict, k: str):
        if len(x[k])==0:
            # the case when a session does not have any connections
            return None # np.zeros(len(bin_edges)-2) 
        else:
            # not counting the bin corresponding to 0 value
            return np.histogram(x[k], bin_edges, density=True)[0]*bin_width
    be_cen_bucketed = get_scalar_dataset_bucketed(
        datasets_bucketed,
        lambda x: hist_helper(x, "graph_edge_be_cen_array")
    )
    fig, axes = plt.subplots(n_buckets,1,figsize=(6,12))
    axes_flat = axes.ravel()
    for i_plot, (chronic_phase, datasets) in enumerate(be_cen_bucketed.items()):
        # datasets is a list of ndarrays; each ndarray is a histogram
        ax = axes_flat[i_plot]
        # filter out the sessions without any monosyn connectivity
        # datasets = list(filter(lambda x: x is not None, datasets))
        datasets_stacked = np.stack(datasets)
        mean_hist = np.mean(datasets_stacked, axis=0)
        sdem_hist = np.std(datasets_stacked, axis=0) / np.sqrt(len(datasets))
        ax.bar(
            bin_centers, mean_hist, 
            yerr=sdem_hist, color="white",edgecolor='black', width=bin_width
        )
        ax.set_title(chronic_phase)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(fig_folder, "be_cen_bucket.png"))
    plt.close()
    test_buckets(be_cen_bucketed, fig_folder, "BeCentralityHistogram")

    # plot microcircuit counting results
    arbitrary_single_animal = list(animal_data_dict.values())[0]
    arbitrary_single_session = list(arbitrary_single_animal.values())[0]
    matched_mc_names = list(arbitrary_single_session["matched_mc_density_dict"].keys())
    print(matched_mc_names)
    fig, axes = plt.subplots(2,3, figsize=(16,8))
    axes_flat = axes.ravel()
    for i_mc, mcname in enumerate(matched_mc_names):
        scalar_dataset_bucketed = get_scalar_dataset_bucketed(
            datasets_bucketed,
            lambda x: x["matched_mc_density_dict"][mcname]
        )
        # plot
        ax = axes_flat[i_mc]
        positions = list(range(len(scalar_dataset_bucketed)))
        # ax.boxplot(
        #     list(dataset_bucketed.values()),
        #     positions=positions,
        #     widths=0.7
        #     )
        data_means = [np.mean(dataset_) for dataset_ in scalar_dataset_bucketed.values()]
        data_sdems = [np.std(dataset_)/np.sqrt(len(dataset_)) for dataset_ in scalar_dataset_bucketed.values()]
        ax.bar(positions, data_means, yerr=data_sdems, color='white', edgecolor='black')
        ax.set_xticks(positions)
        ax.set_xticklabels(list(scalar_dataset_bucketed.keys()))
        # ax.legend()
        ax.set_xlabel("Chronic Phase")
        ax.set_ylabel("Density")
        ax.set_title(mcname)
        # statistical test and export results
        test_buckets(scalar_dataset_bucketed, fig_folder, "count_mc_"+mcname)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(fig_folder, "microcircuits_barplot.png"))
    plt.close()
    

    # connection vertical lengths
    bin_edges = np.linspace(0, 350, 15)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    def hist_helper(x: dict, k: str):
        if len(x[k])==0:
            # the case when a session does not have any connections
            return None # np.zeros(len(bin_edges)-2) 
        else:
            # not counting the bin corresponding to 0 value
            return np.histogram(x[k], bin_edges, density=True)[0]*bin_width
    vd_exc_bucketed = get_scalar_dataset_bucketed(
        datasets_bucketed,
        lambda x: hist_helper(x, "conn_exc_vertdists")
    )
    fig, axes = plt.subplots(n_buckets,1,figsize=(6,12))
    axes_flat = axes.ravel()
    for i_plot, (chronic_phase, datasets) in enumerate(vd_exc_bucketed.items()):
        # datasets is a list of ndarrays; each ndarray is a histogram
        ax = axes_flat[i_plot]
        # filter out the sessions without any monosyn connectivity
        # datasets = list(filter(lambda x: x is not None, datasets))
        datasets_stacked = np.stack(datasets)
        mean_hist = np.mean(datasets_stacked, axis=0)
        sdem_hist = np.std(datasets_stacked, axis=0) / np.sqrt(len(datasets))
        ax.bar(
            bin_centers, mean_hist, 
            yerr=sdem_hist, color="white",edgecolor='black', width=bin_width
        )
        ax.set_title(chronic_phase)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(fig_folder, "vd_exc_bucket.png"))
    plt.close()
    vd_inh_bucketed = get_scalar_dataset_bucketed(
        datasets_bucketed,
        lambda x: hist_helper(x, "conn_inh_vertdists")
    )
    fig, axes = plt.subplots(n_buckets,1,figsize=(6,12))
    axes_flat = axes.ravel()
    for i_plot, (chronic_phase, datasets) in enumerate(vd_inh_bucketed.items()):
        # datasets is a list of ndarrays; each ndarray is a histogram
        ax = axes_flat[i_plot]
        # filter out the sessions without any monosyn connectivity
        # datasets = list(filter(lambda x: x is not None, datasets))
        datasets_stacked = np.stack(datasets)
        mean_hist = np.mean(datasets_stacked, axis=0)
        sdem_hist = np.std(datasets_stacked, axis=0) / np.sqrt(len(datasets))
        ax.bar(
            bin_centers, mean_hist, 
            yerr=sdem_hist, color="white",edgecolor='black', width=bin_width
        )
        ax.set_title(chronic_phase)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(fig_folder, "vd_inh_bucket.png"))
    plt.close()

    # hits distribution
    # fig, axes = plt.subplots(n_animals, len(days_standard), figsize=(20,12), dpi=400)
    # # regions = list(CFG.ShankLoc)[:3]
    # for i_animal_id, animal_id in enumerate(animal_days_dict.keys()):
    #     session_ids = animal_days_dict[animal_id]["session_ids"]
    #     days        = animal_days_dict[animal_id]["days"]
    #     for i_day_id, (day, session_id) in enumerate(zip(days, session_ids)):
    #         ax_ypos = days_standard.index(day)
    #         ax = axes[i_animal_id][ax_ypos]
    #         ddict = animal_data_dict[animal_id][session_id]
    #         h_ = ddict["graph_hits_h_array"]
    #         a_ = ddict["graph_hits_a_array"]
    #         bin_edges = np.arange(0, 1, 11)
    #         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #         h_hist, bin_edges = np.histogram(h_)#, bin_edges)
    #         bin_centers_h = (bin_edges[:-1] + bin_edges[1:]) / 2
    #         a_hist, bin_edges = np.histogram(a_)#, bin_edges)
    #         bin_centers_a = (bin_edges[:-1] + bin_edges[1:]) / 2
    #         ax.bar(bin_centers_h, h_hist, color="orange", width=bin_centers_h[1]-bin_centers_h[0])
    #         ax.bar(bin_centers_a, -a_hist, color="blue", width=bin_centers_a[1]-bin_centers_a[0])
    #         ax.set_title("%s - day %d" % (animal_id, day))
    # plt.tight_layout()
    # plt.suptitle("Chronic HITS distribution for each animal\nOrange: hubs; Blue: authorities")
    # os.makedirs(fig_folder, exist_ok=True)
    # plt.savefig(os.path.join(fig_folder, "HITS.png"))
    # plt.close()

if __name__ == "__main__":
    # data_dicts = []
    # for reldir in cfg.spk_reldirs:
    #     data_folder = os.path.join(cfg.spk_inpdir, reldir)
    #     temp_folder = os.path.join(cfg.mda_tempdir, reldir)
    #     data_dicts.append(read_postproc_data(data_folder, temp_folder, temp_folder))
    parser = argparse.ArgumentParser()
    parser.add_argument('--nocurate', action="store_true", default=False)
    parser.add_argument('--noplot', action="store_true", default=False)
    parser.add_argument('--bucket', action="store_true", default=False)
    parser.add_argument('--exc_only', action="store_true", default=False)
    parser.add_argument('--inh_only', action="store_true", default=False)
    # parser.add_argument('--no_normalize', action="store_true", default=False)
    args = parser.parse_args()
    assert not(args.exc_only and args.inh_only)
    if args.exc_only:
        count_type = 1
    elif args.inh_only:
        count_type = -1
    else:
        count_type = 0
    arg_apply_curation = not(args.nocurate)
    animal_session_id_dict = OrderedDict()
    animal_session_reldir_dict = OrderedDict()
    for reldir in CFG.spk_reldirs:
        a, session_id = reldir.split("/")
        animal_id = "".join(a.split("_")[-1].split("-")).lower()
        if animal_id not in animal_session_id_dict:
            animal_session_id_dict[animal_id] = [session_id]
            animal_session_reldir_dict[animal_id] = [reldir]
        else:
            animal_session_id_dict[animal_id].append(session_id)
            animal_session_reldir_dict[animal_id].append(reldir)
    animal_days_dict = get_valid_days_all_animal(animal_session_id_dict, CFG.strokedays, CFG.days_standard)
    for k, v in animal_days_dict.items():
        print(k, ":", v)
    animal_data_dict = get_connec_data_all_animal(animal_session_id_dict, animal_session_reldir_dict, CFG, arg_apply_curation, count_type)
    print("Plotting")
    if not(args.noplot):
        figfolder = os.path.join(CFG.con_resdir, "plots_20230621")
        os.makedirs(figfolder, exist_ok=True)
        if args.bucket:
            boxplot_conns_for_each_animal(animal_days_dict, animal_data_dict, CFG.days_standard, figfolder)
        else:
            plot_conns_for_each_animal(animal_days_dict, animal_data_dict, CFG.days_standard, figfolder)

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

import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from scipy.io import loadmat

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
        # GW_BETWEENSHANK = 250
        # GH = 30
        max_depth = 30*15
    else:
        # GW_BETWEENSHANK = 300
        # GH = 25
        max_depth = 25*31
    ret["max_depth"] = max_depth
    # ch2shank is the map of each channel to corresponding shank. Each entry is in {0,1,2,3}.
    ch2shank = pd.read_csv(os.path.join(mdaclean_temp_dir, "ch2shank.csv"), header=None).values.squeeze()
    template_waveforms = readmda(os.path.join(mdaclean_temp_dir, "templates_clean.mda")).astype(np.int64)
    map_clean2original = pd.read_csv(os.path.join(mdaclean_temp_dir, "map_clean2original_labels.csv"), header=None).values
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus)
    n_units = pri_ch_lut.shape[0]
    spikewidths = [calc_t2p(template_waveforms[pri_ch_lut[i_], :, i_], session_info["SampleRate"]) for i_ in range(pri_ch_lut.shape[0])]
    spikewidths = np.array([k[0] for k in spikewidths])
    # read putative connectivity
    connecs = pd.read_csv(os.path.join(session_connec_dir, "connecs.csv"), header=None).values
    if apply_curation:
        curate_mask = pd.read_csv(os.path.join(session_connec_dir, "conn_keep_mask.csv"), header=None).values.squeeze()
        curate_mask = (curate_mask>0).astype(bool)
        connecs = connecs[curate_mask, :]
    connecs[:, :2] = connecs[:,:2]-1
    ret["connecs"] = connecs
    ret["conn_cnt_mat"] = np.zeros((4,4), dtype=int)
    ret["conn_cnt_types"] = list(CFG.ShankLoc) # NOTE this assumes the ShankLoc enum type has values from 0 and increments by 1.
    ret["templates_clean"] = template_waveforms
    # unit_shanks = ch2shank[pri_ch_lut[np.arange(n_units)]] # defaq was wrong with me
    unit_shanks = ch2shank[pri_ch_lut]
    # unit_regions = np.array([shank_def[sk_] for sk_ in unit_shanks])
    ret['unit_shanks'] = unit_shanks
    ret['unit_depths'] = geom[pri_ch_lut, 1]

    if not(apply_curation):
        print("  ====%s" % (session_connec_dir))
    
    # for i_connec, (src, snk, contype) in enumerate(connecs[:, :]):
    #     if (count_type!=0 and contype!=count_type):
    #         continue
    #     src_original_id = map_clean2original[src]
    #     snk_original_id = map_clean2original[snk]
    #     src_shanknum    = ch2shank[pri_ch_lut[src]]
    #     snk_shanknum    = ch2shank[pri_ch_lut[snk]]
    #     src_region      = shank_def[src_shanknum] # IntEnum instance
    #     snk_region      = shank_def[snk_shanknum] # IntEnum instance
    #     assert src_region==unit_regions[src] and snk_region==unit_regions[snk]
    #     if src_region == snk_region:
    #         # for connections whose source and target are in the same region, we only consider them if they are in the same shank
    #         if src_shanknum == snk_shanknum:
    #             ret["conn_cnt_mat"][src_region][snk_region] += 1
    #     else:
    #         ret["conn_cnt_mat"][src_region][snk_region] += 1
        
    #     # print cross-shank connection
    #     if not(apply_curation) and src_shanknum != snk_shanknum:
    #         print("    ====i_connec:%3d, Source: %d, Sink: %d" % (i_connec, src, snk))
    # # noramlize by number of edges maximum possible
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
    ret["spikewidths"] = spikewidths

    return ret


def get_connec_data_single_animal(animal_id, session_reldirs, session_ids, cfg_module, apply_curation, count_type):
    # data_dicts = []
    data_dict = OrderedDict()
    for i_session, session_reldir in enumerate(session_reldirs):
        session_spk_dir_   = os.path.join(cfg_module.spk_inpdir, session_reldir)
        mdaclean_temp_dir_ = os.path.join(cfg_module.mda_tempdir, session_reldir)
        monosyn_conn_dir_  = os.path.join(cfg_module.con_resdir, session_reldir)
        data_dict_t = read_postproc_data(session_spk_dir_, mdaclean_temp_dir_, monosyn_conn_dir_, cfg_module.shank_defs[animal_id], apply_curation, count_type)
        data_dict[session_ids[i_session]] = data_dict_t
    return data_dict

def get_connec_data_all_animal(animal_session_id_dict, animal_session_reldir_dict, cfg_module, apply_curation, count_type):
    animal_conn_mats_dict = OrderedDict()
    for animal_id in animal_session_id_dict.keys():
        session_ids = animal_session_id_dict[animal_id]
        session_reldirs = animal_session_reldir_dict[animal_id]
        animal_conn_mats_dict[animal_id] = get_connec_data_single_animal(animal_id, session_reldirs, session_ids, cfg_module, apply_curation, count_type)
    return animal_conn_mats_dict




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



def plot_graphs_for_each_animal(animal_days_dict, animal_data_dict, days_standard, fig_folder):
    """
    animal_days_dict :
        dict["animal_id"]["days"] is the list of valid days for specified animal
        dict["animal_id"]["session_ids"] is the list of session ids for corresponding days for specified animal
    animal_data_dict : dict["animal_id"]["session_id"] is the data dict from corresponding session.
    """
    n_animals = len(animal_data_dict)
    fig, axes = plt.subplots(n_animals, len(days_standard), figsize=(20,12), dpi=400)
    # axes_flat = axes.flatten()
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    # regions = list(CFG.ShankLoc)[:3]
    for i_animal_id, animal_id in enumerate(animal_days_dict.keys()):
        session_ids = animal_days_dict[animal_id]["session_ids"]
        days        = animal_days_dict[animal_id]["days"]
        for i_day_id, (day, session_id) in enumerate(zip(days, session_ids)):
            ax_ypos = days_standard.index(day)
            ax = axes[i_animal_id][ax_ypos]
            ddict = animal_data_dict[animal_id][session_id]
            edge_table  = ddict["connecs"]
            unit_shanks = ddict['unit_shanks']
            unit_depths  = ddict['unit_depths'] / ddict['max_depth']
            n_units     = len(unit_shanks)
            g = utils_graph.create_graph(n_units, edge_table)
            utils_graph.plot_graph(g, unit_shanks, unit_depths, ax)
            ax.set_title("%s - day %d" % (animal_id, day))
    plt.tight_layout()
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, "graphs.png"))
    plt.close()


if __name__ == "__main__":
    # data_dicts = []
    # for reldir in cfg.spk_reldirs:
    #     data_folder = os.path.join(cfg.spk_inpdir, reldir)
    #     temp_folder = os.path.join(cfg.mda_tempdir, reldir)
    #     data_dicts.append(read_postproc_data(data_folder, temp_folder, temp_folder))
    parser = argparse.ArgumentParser()
    parser.add_argument('--nocurate', action="store_true", default=False)
    parser.add_argument('--noplot', action="store_true", default=False)
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
    animal_conn_mats_dict = get_connec_data_all_animal(animal_session_id_dict, animal_session_reldir_dict, CFG, arg_apply_curation, count_type)
    if not(args.noplot):
        plot_graphs_for_each_animal(animal_days_dict, animal_conn_mats_dict, CFG.days_standard, CFG.con_resdir)


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
from time import time

import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from scipy.io import loadmat

sys.path.append("../Spike-Sorting-Code/preprocess_rhd")
from utils.read_mda import readmda

# sys.path.append("../Spike-Sorting-Code/post_msort_processing/utils")
# from waveform_metrics import calc_t2p
from synchrony import get_synchrony_matrix
import config as CFG


def process_postproc_data(session_spk_dir: str, mdaclean_temp_dir: str, session_connec_dir: str, shank_def: dict, rng: np.random.Generator) -> dict:
    """ process post processed data
    Assumes the 'label' field in combine_metrics_new.json is 1..n_clus_uncurated
    """
    print(session_connec_dir)
    ### read clustering metrics file
    with open(os.path.join(session_spk_dir, "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)
    spike_counts = np.array([c["metrics"]["num_events"] for c in x["clusters"]])
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
    map_clean2original_labels = pd.read_csv(os.path.join(mdaclean_temp_dir, "map_clean2original_labels.csv"), header=None).values.squeeze()
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus)
    spike_counts = spike_counts[map_clean2original_labels-1].astype(np.int64)
    ret["spike_counts"] = spike_counts

    mono_res_mat = loadmat(os.path.join(session_connec_dir, "mono_res.cellinfo.mat"))
    ccg_binsize_ms = mono_res_mat["mono_res"][0][0]['binSize'][0][0]*1000
    ccg_nbins = mono_res_mat["mono_res"][0][0]['ccgR'].shape[0]
    ret["ccg_bincenters_ms"] = (np.arange(ccg_nbins)-(ccg_nbins//2))*ccg_binsize_ms
    ret["ccg"] = mono_res_mat["mono_res"][0][0]['ccgR'].transpose([1,2,0]).copy(order="C") # force contiguous (n_units, n_units, ccg_nbins) memory layout
    synchrony_savepath = os.path.join(session_connec_dir, "synchrony.npz")
    if not os.path.exists(synchrony_savepath):
        print("Start calculating synchrony")
        ts = time()
        sm, phi, plo = get_synchrony_matrix(ret["ccg"], ret["spike_counts"], ret["ccg_bincenters_ms"], 15, 200, rng)
        te = time()-ts
        print("Synchrony calculation time:", te)
        ret["synchrony_val_matrix"] = sm
        ret["synchrony_phi_matrix"] = phi
        ret["synchrony_plo_matrix"] = plo
        np.savez(
            synchrony_savepath,
            synchrony_val_matrix=sm,
            synchrony_phi_matrix=phi,
            synchrony_plo_matrix=plo
        )

        plt.figure()
        plt.subplot(1,2,1)
        z1 = plt.imshow(sm, cmap="hot", vmin=0.0, vmax=1.0)
        plt.colorbar(z1, ax=plt.gca())
        plt.title("Synchrony matrix")
        plt.subplot(1,2,2)
        z2 = plt.imshow(phi, cmap="hot", vmin=0.0, vmax=1.0)
        plt.colorbar(z2, ax=plt.gca())
        plt.title("Percentage of randomly shuffled\nsynchrony larger than observed")
        plt.savefig(os.path.join(session_connec_dir, "synchrony.png"))
        # plt.show()
        plt.close()
    else:
        tmp = np.load(synchrony_savepath)
        ret["synchrony_val_matrix"] = tmp["synchrony_val_matrix"]
        ret["synchrony_phi_matrix"] = tmp["synchrony_phi_matrix"]
        ret["synchrony_plo_matrix"] = tmp["synchrony_plo_matrix"]
    # TODO 1. bucket the synchrony values by shank regions
    # TODO 2. get masked array of only the significant synchronies
    # TODO 3. get average synchrony matrix of (n_regions, n_regions)
    unit_shanks = ch2shank[pri_ch_lut] # (n_units,)
    n_units = len(unit_shanks)
    region_synchronies = np.zeros((3,3))
    # for (shankname, units_set) in groupby(sorted(np.arange(n_units), key=shank_def.__getitem__), key=shank_def.__getitem__):
    #     units_ = list(units_set)

    return ret


def get_connec_data_single_animal(animal_id, session_reldirs, session_ids, cfg_module, apply_curation):
    # data_dicts = []
    region_conn_mats_dict = OrderedDict()
    for i_session, session_reldir in enumerate(session_reldirs):
        session_spk_dir_   = os.path.join(cfg_module.spk_inpdir, session_reldir)
        mdaclean_temp_dir_ = os.path.join(cfg_module.mda_tempdir, session_reldir)
        monosyn_conn_dir_  = os.path.join(cfg_module.con_resdir, session_reldir)
        data_dict = process_postproc_data(session_spk_dir_, mdaclean_temp_dir_, monosyn_conn_dir_, cfg_module.shank_defs[animal_id], apply_curation)

        region_conn_mats_dict[session_ids[i_session]] = data_dict["synchrony_val_matrix"]
    return region_conn_mats_dict

def get_connec_data_all_animal(animal_session_id_dict, animal_session_reldir_dict, cfg_module, apply_curation):
    animal_conn_mats_dict = OrderedDict()
    for animal_id in animal_session_id_dict.keys():
        session_ids = animal_session_id_dict[animal_id]
        session_reldirs = animal_session_reldir_dict[animal_id]
        animal_conn_mats_dict[animal_id] = get_connec_data_single_animal(animal_id, session_reldirs, session_ids, cfg_module, apply_curation)
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



def plot_conns_for_each_animal(animal_days_dict, animal_conn_mats_dict, days_standard, fig_folder):
    """
    animal_days_dict :
        dict["animal_id"]["days"] is the list of valid days for specified animal
        dict["animal_id"]["session_ids"] is the list of session ids for corresponding days for specified animal
    animal_conn_mats_dict : dict["animal_id"]["session_id"][src_region][snk_region] is the number of conns; the matrix is symmetrical if undirected.
    """
    fig, axes = plt.subplots(2,3, figsize=(16,8))
    axes_flat = axes.flatten()
    regions = list(CFG.ShankLoc)[:3]
    k = 0
    for i in range(3):
        src_region = regions[i]
        for j in range(i, 3):
            snk_region = regions[j]
            ax = axes_flat[k]
            k += 1
            for animal_id in animal_days_dict.keys():
                days = animal_days_dict[animal_id]["days"]
                session_ids = animal_days_dict[animal_id]["session_ids"]
                dataset = [animal_conn_mats_dict[animal_id][session_id][src_region][snk_region] for session_id in session_ids]
                ax.plot(days, dataset, marker='x', markersize=2.0, linewidth=1.0, label=animal_id)
            ax.set_xticks(days_standard)
            ax.legend()
            ax.set_xlabel("Day")
            ax.set_ylabel("#Conns")
            ax.set_title("%s<-->%s"% (src_region.name, snk_region.name))
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, "raw.png"))
    plt.close()


if __name__ == "__main__":
    # data_dicts = []
    # for reldir in cfg.spk_reldirs:
    #     data_folder = os.path.join(cfg.spk_inpdir, reldir)
    #     temp_folder = os.path.join(cfg.mda_tempdir, reldir)
    #     data_dicts.append(read_postproc_data(data_folder, temp_folder, temp_folder))
    parser = argparse.ArgumentParser()
    # parser.add_argument('--nocurate', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--noplot', action="store_true", default=False)
    args = parser.parse_args()
    # arg_apply_curation = not(args.nocurate)
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
    rng = np.random.default_rng(args.seed)
    animal_conn_mats_dict = get_connec_data_all_animal(animal_session_id_dict, animal_session_reldir_dict, CFG, rng)
    if not(args.noplot):
        plot_conns_for_each_animal(animal_days_dict, animal_conn_mats_dict, CFG.days_standard, CFG.con_resdir)


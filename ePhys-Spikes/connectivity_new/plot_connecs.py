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
import config as cfg

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.size']=30

SHANK_SPACING = 300


CELLTYPE_PYRAMIDAL = 0
CELLTYPE_NARROW_INTER = 1
CELLTYPE_WIDE_INTER = 2
CELLTYPES = {
    CELLTYPE_PYRAMIDAL: "Pyramidal",
    CELLTYPE_NARROW_INTER: "Narrow Inter",
    CELLTYPE_WIDE_INTER: "Wide Inter"
}

SHANK_SWAP_BC = False

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
    # get shank# for each channel
    geom = pd.read_csv(os.path.join(session_spk_dir, "geom.csv"), header=None).values
    if session_info['ELECTRODE_2X16']:
        GW_BETWEENSHANK = 250
    else:
        GW_BETWEENSHANK = 300
    ch2shank = pd.read_csv(os.path.join(mdaclean_temp_dir, "ch2shank.csv"), header=None).values
    template_waveforms = readmda(os.path.join(mdaclean_temp_dir, "templates_clean.mda")).astype(np.int64)
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus)
    # read putative connectivity
    connecs = pd.read_csv(os.path.join(session_connec_dir, "connecs.csv"), header=None).values
    connecs[:, :2] = connecs[:,:2]-1
    ret["connecs"] = connecs
    n_c_wishank = 0
    n_c_bwshank = 0
    for src, snk in connecs[:, :2]:
        if (ch2shank[pri_ch_lut[src]]) == (ch2shank[pri_ch_lut[snk]]):
            n_c_wishank += 1
        else:
            n_c_bwshank += 1
    ret["n_con_bw_shank"] = n_c_bwshank
    ret["n_con_wi_shank"] = n_c_wishank
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


def proc_main(spk_dir_root, mda_tempdir_root, connec_dir_root, spk_reldirs, mda_reldirs):
    animal_dir = mda_reldirs[0].split("/")[0]
    pp_dicts = []
    for mda_reldir, spk_reldir in zip(mda_reldirs, spk_reldirs):
        session_spk_dir = os.path.join(spk_dir_root, spk_reldir)
        session_mda_dir = os.path.join(mda_tempdir_root, mda_reldir)
        session_con_dir = os.path.join(connec_dir_root, mda_reldir)
        pp_dict = read_postproc_data(session_spk_dir, session_mda_dir, session_con_dir)
        pp_dict['session_datestr'] = get_datetime_str(mda_reldir.split("/")[-1])
        pp_dict['session_datetime'] = get_datetime(mda_reldir.split("/")[-1])
        pp_dicts.append(pp_dict)
        # break

    # session_datestrs = [d['session_idstr'].replace('\\', '/').split('/')[-1] for d in pp_dicts]
    # session_dates    = [datetime.strptime(ds, "%Y-%m-%d") for ds in session_datestrs]
    # session_relative_days = np.array([round((session_date-session_dates[0]).total_seconds()/24/3600) for session_date in session_dates])
    # stroke_session_idx = np.where([p['is_strokeday'] for p in pp_dicts])[0][0]
    # session_relative_days = session_relative_days - session_relative_days[stroke_session_idx]

    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams['font.size']=22

    # plot total firing rates for each shank shank
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    session_datestrs = [d['session_datestr'] for d in pp_dicts]
    session_datetimes = [d['session_datetime'] for d in pp_dicts]
    session_ncbwshank = [d['n_con_bw_shank'] for d in pp_dicts]
    session_ncwishank = [d['n_con_wi_shank'] for d in pp_dicts]
    ax.plot(session_datetimes, session_ncbwshank, label="# connections b/w channel shank")
    ax.plot(session_datetimes, session_ncwishank, label="# connections w/i channel shank")
    ax.set_xticks(session_datetimes)
    ax.set_xticklabels(session_datestrs, rotation=90)
    ax.set_xlabel("Time (day)")
    ax.set_ylabel("# connectivties")
    # plt.show()
    ax.legend()
    plt.savefig(os.path.join(connec_dir_root, animal_dir, "wi-vs-bw.png"))
    plt.show()
    # plt.close()

import config as cfg
proc_main(
    cfg.spk_inpdir,
    cfg.mda_tempdir,
    cfg.con_resdir,
    cfg.spk_reldirs,
    cfg.mda_reldirs
)

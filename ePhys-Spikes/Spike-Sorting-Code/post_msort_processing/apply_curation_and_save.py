'''
For cleaning up the mda files after rejection but before merging
The merging code shouldn't use results from this code
This is just for creating inputs CellExplorer functions right after rejection
'''
import os, sys
from time import time
from copy import deepcopy
import gc
import json
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

sys.path.append("../preprocess_rhd")
from utils.read_mda import readmda
from utils.mdaio import writemda64


def reorder_mda_arrs(
    all_waveforms_by_cluster ,clus_locations : np.ndarray, firings_full : np.ndarray,
    templates_full : np.ndarray, clean_mask : np.ndarray, prim_channels_full: np.ndarray
    ):
    """reorder mda arrays
    Parameters
    ----------
    all_waveforms_by_cluster: nested list 
    clus_locations : (n_clus_raw,2) <float> 
    firings_full : (3, n_clus_raw) <int> [primary channel; event time(sample); unit label]
    templates_full : (n_chs, waveform_len, n_clus_raw)
    clean_mask : (n_chs,) <boolean>
    prim_channels_full : (n_clus_raw,) <int> elements are positive (channel indexes from 1)
    """

    n_clus = templates_full.shape[2]
    n_clus_clean = np.sum(clean_mask)
    # (1) mapping from original to curated
    map_clean2original_labels = np.where(clean_mask)[0]+1 # "labels" means base-1
    map_original2clean_labels = -1*np.ones(n_clus)
    map_original2clean_labels[clean_mask] = np.arange(n_clus_clean)+1
    map_original2clean_labels = map_original2clean_labels.astype(int)
    # (2) reorganize templates.mda
    templates_clean = templates_full[:, :, clean_mask]
    # (3) reorganize firings.mda
    firings_clean = firings_full.copy()
    # (3.1) reorganize unit labels
    tmp_labels_old = firings_full[2,:] # get raw unit labels
    if not np.all(tmp_labels_old>0):
        raise ValueError("Labels should start from 1")
    tmp_labels_new = map_original2clean_labels[tmp_labels_old-1] # map from raw to clean unit labels 
    firings_clean[2,:] = tmp_labels_new # update firings data structure
    spikes_keep_mask = firings_clean[2,:]!=-1 # decide which events to keep
    firings_clean = firings_clean[:, spikes_keep_mask] # clean up
    # (3.2) reorganize unit primary channels
    prim_channels_clean = prim_channels_full[clean_mask]
    tmp_prichs_new = prim_channels_clean[firings_clean[2,:]-1]
    firings_clean[0,:] = tmp_prichs_new
    clus_locations_new = clus_locations[clean_mask,:]       # cleaning up the cluster_locations.csv file here
    # Cleaning file all_waveforms_by_cluster.npz (This file saves all clusters all spike waveforms)
    waveforms_all_dict = OrderedDict()
    nTemplates = clus_locations.shape[0]
    i_clus_real = 0
    for i_clus in range(nTemplates):
        if clean_mask[i_clus]:
            waveforms_this_cluster = all_waveforms_by_cluster['clus%d'%(i_clus+1)]
            waveforms_all_dict['clus%d'%(i_clus_real+1)] = waveforms_this_cluster
            i_clus_real += 1
   
    
    
    return waveforms_all_dict,clus_locations_new,firings_clean, templates_clean, map_clean2original_labels
    

def clean_mdas(msort_path, postproc_path, mda_savepath):
    """ read post processed data
    !!! Different from the case of processing entire session,
    In this case,  one segment may miss entirely the firing of a neuron, causing 
        (1) the metrics.json to be shorter than the true #units
        (2) corresponding position at template.mda to be NaN
    So we keep track of the "clus_labels" from metrics.json
    And reconstruct the metrics of full length. The missing units will be marked -1 isolation and 999 noise overlap
    And corresponding template to be Zero
    """
    
    ### read clustering metrics file 
    # const_SEGMENT_LEN = 3600 # seconds
    with open(os.path.join(msort_path, "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)
    # read firing stamps, template and continuous waveforms from MountainSort outputs and some processing
    firings = readmda(os.path.join(msort_path, "firings.mda")).astype(np.int64)
    template_waveforms = readmda(os.path.join(msort_path, "templates.mda")).astype(np.float64)
    clus_locations = pd.read_csv(os.path.join(msort_path, 'clus_locations.csv'),header = None)
    all_waveforms_by_cluster = np.load(os.path.join(msort_path,'all_waveforms_by_cluster.npz'))
    clus_locationsA = clus_locations.to_numpy()
    n_clus = template_waveforms.shape[2]
    # set nan to zero just in case some units don't fire during the segment resulting in nan 
    template_waveforms = np.nan_to_num(template_waveforms, nan=0.0)
    # read cluster metrics to find out which units did not spike during the segment
    clus_metrics_list = x['clusters']
    clus_labels = np.array([k['label'] for k in clus_metrics_list])
    peak_snr_short = np.array([k['metrics']['peak_snr'] for k in clus_metrics_list])
    peak_snr = np.ones(n_clus, dtype=float)*(-1)
    peak_snr[clus_labels-1] = peak_snr_short
    spiking_mask = (peak_snr>=0)
    # read rejection mask
    # accept_mask = np.load(os.path.join(postproc_path, "cluster_rejection_mask.npz"))['single_unit_mask'].astype(bool)
    accept_mask = pd.read_csv(os.path.join(postproc_path, 'accept_mask.csv') , header = None).values.squeeze().astype(bool)
    positive_mask = pd.read_csv(os.path.join(postproc_path, "positive_mask.csv"), header=None).values.squeeze().astype(bool)
    # clusters to keep: both (1) spiking and (2) accepted by curation criteria
    clean_mask = np.logical_and(spiking_mask, accept_mask)
    clean_mask = np.logical_and(clean_mask, np.logical_not(positive_mask))
    # get primary channel; channel index starts from 0 here
    pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus)
    # do the actual cleaning up
    waveforms_all_dict, clus_locations_clean, firings_clean, templates_clean, map_clean2original_labels = reorder_mda_arrs(
        all_waveforms_by_cluster,clus_locationsA,firings, template_waveforms, clean_mask, pri_ch_lut+1
        )
    writemda64(firings_clean, os.path.join(mda_savepath, "firings_clean.mda"))
    writemda64(templates_clean, os.path.join(mda_savepath, "templates_clean.mda"))
    pd.DataFrame(clus_locations_clean).to_csv(os.path.join(mda_savepath,'clus_locations_clean.csv'),index = False ,header = False)
    np.savez(os.path.join(mda_savepath, "all_waveforms_by_cluster_clean.npz"), **waveforms_all_dict)
    
    pd.DataFrame(data=map_clean2original_labels).to_csv(
        os.path.join(mda_savepath, "map_clean2original_labels.csv"), 
        index=False, header=False
        )



def clean_mdas_main(spk_folders, clean_mda_output_folders):

    for msort_folder, clean_folder in zip(spk_folders, clean_mda_output_folders):
        os.makedirs(clean_folder, exist_ok=True)
        clean_mdas(msort_folder, msort_folder, clean_folder)

"""
Functions for performing PCA on waveforms. Based on SpikeInterface implementation
https://github.com/SpikeInterface/spikeinterface/blob/0247c57f5e510e630e1fbc59060c2e75fde3b3b6/spikeinterface/postprocessing/principal_component.py#L404
"""

import itertools

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KDTree

def sort_and_groupby(iterable, *, keyfunc):
    for (kname, elems) in itertools.groupby(sorted(iterable, key=keyfunc), key=keyfunc):
        yield kname, list(elems)

def helper_partial_fit(model_identifier, data_identifier, result_container):
    raise NotImplementedError()

def compute_pca_by_channel(spk_waves_all, spk_labels_all, valid_channels_per_unit, mppath, n_components=3, n_jobs=1, whiten=True):
    """
    Parameters
    --------
    spk_waves_all : (n_spikes, len_waveform, n_channels) all spike waveforms 
    spk_labels_all : (n_spikes, ) spike labels start from 0 
    valid_channels_per_unit : (n_units, n_neighboorhood_channels) (valid channels to compute pca for each unit) (channels start from 0)

    Retunrs
    --------
    pcs : (n_spikes, n_components, n_neighboorhood_channels)
    valid_channels_per_unit is through
    """
    pca_models = {}
    assert spk_labels_all.shape[0] == spk_waves_all.shape[0]
    n_spikes = spk_labels_all.shape[0]
    k_chs = valid_channels_per_unit.shape[1]
    # job_specs = [] # (identifier for model, data[m_spikes, len_waveform], channel)

    print("Computing PCA: fitting in increments")
    # first run - perform incremental fitting
    for spk_label0, spk_inds in sort_and_groupby(spk_labels_all, keyfunc=lambda k: k):
        channels_oi = valid_channels_per_unit[spk_label0]
        for ch_idx in channels_oi:
            waveforms_tmp = spk_waves_all[spk_inds, :, ch_idx]
            if ch_idx not in pca_models:
                pca_models[ch_idx] = IncrementalPCA(n_components=n_components, whiten=whiten)
            if n_jobs <= 1:
                pca_models[ch_idx].partial_fit(waveforms_tmp)
            else:
                # TODO use joblibs delayed and Parallel
                # fit the data (concataneted and mped) in each channel in parallel
                raise NotImplementedError()
            
    print("Computing PCA: applying transform")
    # second run - perform PCA transfom
    pcs = np.memmap(mppath, shape=(n_spikes, n_components, k_chs), dtype=np.float32, mode="w+")
    for spk_label0, spk_inds in sort_and_groupby(spk_labels_all, keyfunc=lambda k: k):
        channels_oi = valid_channels_per_unit[spk_label0]
        for ch_idx in channels_oi:
            waveforms_tmp = spk_waves_all[spk_inds, :, ch_idx]
            pcs[spk_inds, :, ch_idx] = pca_models[ch_idx].transform(waveforms_tmp).astype(np.float32)
    pcs.flush()
    return pcs

def compute_pca_single_channel(spk_waves_all, spk_labels_all, prim_channels, mppath, n_components=3, n_jobs=1, whiten=True):
    """
    Parameters
    --------
    spk_waves_all : (n_spikes, len_waveform) all spike waveforms 
    spk_labels_all : (n_spikes, ) spike labels start from 0 
    valid_channels_per_unit : (n_units, n_neighboorhood_channels) (valid channels to compute pca for each unit) (start from 0)

    Retunrs
    --------
    pcs : (n_spikes, n_components, n_neighboorhood_channels)
    valid_channels_per_unit is through
    """
    pca_models = {}
    assert spk_labels_all.shape[0] == spk_waves_all.shape[0], "%s vs %s" % (spk_labels_all.shape, spk_waves_all.shape)
    n_spikes = spk_labels_all.shape[0]
    # job_specs = [] # (identifier for model, data[m_spikes, len_waveform], channel)

    print("Computing PCA: fitting in increments")
    # first run - perform incremental fitting
    for spk_label0, spk_inds in sort_and_groupby(spk_labels_all, keyfunc=lambda k: k):
        ch_idx = prim_channels[spk_label0]
        waveforms_tmp = spk_waves_all[spk_inds, :]
        if ch_idx not in pca_models:
            pca_models[ch_idx] = IncrementalPCA(n_components=n_components, whiten=whiten)
        if n_jobs <= 1:
            pca_models[ch_idx].partial_fit(waveforms_tmp)
        else:
            # TODO use joblibs delayed and Parallel
            # fit the data (concataneted and mped) in each channel in parallel
            raise NotImplementedError()
            
    print("Computing PCA: applying transform")
    # second run - perform PCA transfom
    # pcs = np.memmap(mppath, shape=(n_spikes, n_components, 1), dtype=np.float32, mode="w+")
    pcs = np.empty(shape=(n_spikes, n_components, 1), dtype=np.float32)
    for spk_label0, spk_inds in sort_and_groupby(spk_labels_all, keyfunc=lambda k: k):
        ch_idx = prim_channels[spk_label0]
        waveforms_tmp = spk_waves_all[spk_inds, :]
        pcs[spk_inds, :, 0] = pca_models[ch_idx].transform(waveforms_tmp).astype(np.float32)
    # pcs.flush()
    return pcs


def get_nearest_neighboors(prim_locations, channel_locations, k_chs):
    """
    Return shape (n_units, k_chs)
    """
    kt = KDTree(channel_locations, metric="euclidean")
    nb_ch_idss = kt.query(prim_locations, k=k_chs, return_distance=False)
    return nb_ch_idss

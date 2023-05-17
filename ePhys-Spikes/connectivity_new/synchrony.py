import gc

import numpy as np

def _get_synchrony_matrix(ccg_tensor, nspikes_vector, ccg_bincenters, tau):
    """
    Parameters
    --------
    ccg_tensor: (n_units, n_units, n_bins)
    nspikes_vector: (n_units,) of total spike counts for each unit
    ccg_bincenters: (n_bins,) of all the bin centers in millisecond
    tau : [float] the range of time within which to count for synchrony

    Returns : synchrony matrix of size (n_units, n_units)
    """
    ccg_mask = ((ccg_bincenters>-tau) & (ccg_bincenters<tau))
    sync_num = np.sum(ccg_tensor[:,:,ccg_mask], axis=-1)
    nspk_sq  = nspikes_vector**2
    sync_denom = np.sqrt(nspk_sq[:,None]+nspk_sq[None,:])
    return sync_num/sync_denom

def get_synchrony_matrix(ccg_tensor, nspikes_vector, ccg_bincenters, tau,
                         n_shuffles, rng=None):
    """
    ccg_tensor: (n_units, n_units, n_bins)
    nspikes_vector: (n_units,) of total spike counts for each unit
    ccg_bincenters: (n_bins,) of all the bin centers in millisecond
    tau : [float] the range of time within which to count for synchrony
    n_shuffles : int : how many shuffles
    rng : np.random.Generator type

    Returns :
    sm_real : (n_units, n_units), observed synchrony matrix
    p_hi : (n_units, n_units) the probability of a shuffled CCG[i,j]
        displaying synchrony not less than sm_real[i,j],
        this is an indicator for the significane of a high synchrony.
    p_lo : (n_units, n_units) similar to p_hi but indicates significance of
        low synchrony values.
    """
    ccg_mask = ((ccg_bincenters>-tau) & (ccg_bincenters<tau))
    nspk_sq  = nspikes_vector**2
    sync_denom = np.sqrt(nspk_sq[:,None]+nspk_sq[None,:])

    sm_real = np.sum(ccg_tensor[:,:,ccg_mask], axis=-1) / sync_denom

    if n_shuffles < 1:
        return sm_real
    sm_all = []
    assert rng is not None, "Must pass an np.random.Generator instance"
    for i_shuffle in range(n_shuffles):
        ccg_xuf = ccg_tensor.copy()
        gc.collect()
        rng.shuffle(ccg_xuf, axis=2) # each unit-pair CCG is being shuffled
                                     # in the same way, which should be ok
        smx = np.sum(ccg_xuf[:,:,ccg_mask], axis=-1) / sync_denom
        sm_all.append(smx)
    sm_all = np.stack(sm_all, axis=0) # (n_shuffles, n_units, n_units)
    p_hi = np.sum( (sm_all>=(sm_real[None,:,:])), axis=0) / n_shuffles
    p_lo = np.sum( (sm_all<=(sm_real[None,:,:])), axis=0) / n_shuffles
    return sm_real, p_hi, p_lo


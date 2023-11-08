#! /bin/env python
#
# Michael Gibson 27 April 2015

# modified by jz103 Dec 20 2021

import numpy as np
import scipy.signal as signal

def notch_filter(input, fSample, fNotch, Bandwidth=None, Q=20):
    """Implements a notch filter (e.g., for 50 or 60 Hz) on vector 'input'.

    fSample = sample rate of data (input Hz or Samples/sec)
    fNotch = filter notch frequency (input Hz)
    Bandwidth = notch 3-dB bandwidth (input Hz).  A bandwidth of 10 Hz is
    recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Example:  If neural data was sampled at 30 kSamples/sec
    and you wish to implement a 60 Hz notch filter:

    out = notch_filter(input, 30000, 60, 10);
    
    If Bandwidth is given, then Q is ignored.
    Defaults to Bandwith=None and Q=20
    """
    if Bandwidth is not None:
        Q = fNotch/Bandwidth
    b, a = signal.iirnotch(fNotch, Q, fs=fSample)
    out = signal.filtfilt(b, a, input, axis=-1)
    
    return out

import json
import os
import contextlib
import sys
import itertools

from .utils_crossmod import *
from .read_mda import readmda
from .read_stimtxt import read_stimtxt
from .csd import inverse_csd
from .lowess import lowess

@contextlib.contextmanager
def logtofile(filename="/dev/null"):
    oldstdout = sys.stdout
    f_stdout = open(filename, "w")
    sys.stdout = f_stdout
    yield
    f_stdout.close()
    sys.stdout = oldstdout

def sort_and_groupby(iterable, *, keyfunc):
    for (kname, elems) in itertools.groupby(sorted(iterable, key=keyfunc), key=keyfunc):
        yield kname, list(elems)


def reject_trials_by_ephys(data, vth, duration_sec, fs):
    '''Assumes data is (n_trials, n_channels, n_samples)'''
    start = int(duration_sec[0]*fs)
    end   = int(duration_sec[1]*fs)
    return (np.max(data[:,:,start:end], axis=(1,2)) <= vth)


def gkern2d(l, sig=1.):
    """ https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
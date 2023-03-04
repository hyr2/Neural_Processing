import json
import os
import contextlib
import sys
import itertools

from .utils_crossmod import *
from .read_mda import readmda
from .read_stimtxt import read_stimtxt
from .csd import inverse_csd

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
import json
import os
import contextlib
import sys

from .utils_crossmod import *
from .read_mda import readmda
from .read_stimtxt import read_stimtxt

@contextlib.contextmanager
def logtofile(filename="/dev/null"):
    oldstdout = sys.stdout
    f_stdout = open(filename, "w")
    sys.stdout = f_stdout
    yield
    f_stdout.close()
    sys.stdout = oldstdout
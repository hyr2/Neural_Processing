import os

import numpy as np
from scipy.io import savemat

from utils.read_mda import readmda

def mda2mat(mda_path, mat_path):
    mda = readmda(mda_path)
    savemat(mat_path, {"mdarray":mda})


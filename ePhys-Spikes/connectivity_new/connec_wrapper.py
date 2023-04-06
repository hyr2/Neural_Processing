""" 
Calls CellExplorer connectivity function from Python. The input format fits barrel-stroke project
3/29/2023 jiaaoZ
"""
import os, sys
import re
import shutil
import subprocess
import json

import numpy as np
import pandas as pd

from clean_firings_mda import clean_mdas
sys.path.append("../Spike-Sorting-Code/preprocess_rhd")
from utils.mdaio import readmda, writemda64


def connec_wrapper_main(spk_dir_root, mda_tempdir_root, connec_outputdir_root, spk_reldirs, mda_reldirs):
    
    # prepare inputs for each given session
    list_fs = []
    for spk_reldir, mda_reldir in zip(spk_reldirs, mda_reldirs):
        session_spk_dir = os.path.join(spk_dir_root, spk_reldir)
        session_mda_tempdir = os.path.join(mda_tempdir_root, mda_reldir)
        os.makedirs(session_mda_tempdir, exist_ok=True)
        # clean_mda stores cleaned firings.mda and templates.mda at the temp folder
        clean_mdas(session_spk_dir, session_spk_dir, session_mda_tempdir)
        with open(os.path.join(session_spk_dir, "pre_MS.json"), "r") as f:
            session_info = json.load(f)
        list_fs.append(session_info["SampleRate"])
        # get shank# for each channel
        geom = pd.read_csv(os.path.join(session_spk_dir, "geom.csv"), header=None).values
        if session_info['ELECTRODE_2X16']:
            GW_BETWEENSHANK = 250
        else:
            GW_BETWEENSHANK = 300
        ch2shank = np.array([(geom[ch_id, 0]//GW_BETWEENSHANK) for ch_id in range(geom.shape[0])])
        pd.DataFrame(data=ch2shank).to_csv(
            os.path.join(session_mda_tempdir, "ch2shank.csv"), 
            index=False, header=False
        )

    # set MATLAB command line arguments
    matlab_argin1 = "{"
    matlab_argin2 = "{"
    for fs_, mda_reldir in zip(list_fs, mda_reldirs):
        session_mda_tempdir = os.path.join(mda_tempdir_root, mda_reldir)
        matlab_argin1 += "\'%s\'"%(session_mda_tempdir)
        matlab_argin1 += ", "
        matlab_argin2 += "%.1f, " % (fs_)
    matlab_argin1 += "}"
    matlab_argin2 += "}"
    
    matlab_args = ["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r"]
    matlab_cmd = "cd utils_matlab; "
    matlab_cmd += "process_segs_batch(%s, %s); exit;" % (matlab_argin1, matlab_argin2)
    matlab_args.append(matlab_cmd)
    print(matlab_args)
    # run MATLAB
    cproc = subprocess.run(matlab_args)
    print("Return code:", cproc.returncode)

    # bring the output back
    for mda_reldir in mda_reldirs:
        session_mda_tempdir = os.path.join(mda_tempdir_root, mda_reldir)
        session_output_dir = os.path.join(connec_outputdir_root, mda_reldir)
        os.makedirs(session_output_dir, exist_ok=True)
        assert os.path.exists(session_mda_tempdir)
        shutil.copy2(os.path.join(session_mda_tempdir, "connecs.csv"), os.path.join(session_output_dir, "connecs.csv"))
        shutil.copy2(os.path.join(session_mda_tempdir, "mono_res.cellinfo.mat"), os.path.join(session_output_dir, "mono_res.cellinfo.mat"))

import config as cfg
connec_wrapper_main(
    cfg.spk_inpdir,
    cfg.mda_tempdir,
    cfg.con_resdir,
    cfg.spk_reldirs,
    cfg.mda_reldirs
)
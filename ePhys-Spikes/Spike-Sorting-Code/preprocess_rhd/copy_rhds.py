import shutil
import os
import glob
import warnings

rawfolder = "/media/G1/xl_neurovascular_coupling/November_2021 - NVC/RH-9/"
locfolder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/data/RH-9"
if not os.path.exists(locfolder):
    os.makedirs(locfolder)


sessions = [
    "22-12-14",
    "22-12-15",
    "22-12-16",
    "22-12-17"
    # "22-12-19",
    # "22-12-23",
    # "22-12-28",
    # "23-01-04",
    # "23-01-11",
    # "23-01-18",
    # "23-01-25",
    # "23-02-01",
    # "23-02-08",
]


for session in sessions:
    print(session)
    ephys_path_pattern = os.path.join(rawfolder, session, "ePhys", "**/*.rhd")
    ephys_rhd_paths = glob.glob(ephys_path_pattern, recursive=True)

    stimtxt_path_pattern = os.path.join(rawfolder, session, "**/whisker_stim.txt")
    stimtxt_paths = glob.glob(stimtxt_path_pattern, recursive=True)
    if len(stimtxt_paths)>0:
        stimtxt_path = stimtxt_paths[0]
        print ("    %s"%stimtxt_path)
    else:
        warnings.warn("Session %s has no `whisker_stim.txt`!"%(session))
        # raise FileNotFoundError("Session %s has no `whisker_stim.txt`!"%(session))

    locpath = os.path.join(locfolder, session)
    if not os.path.exists(locpath):
        os.makedirs(locpath)
    
    for rhdfile in ephys_rhd_paths:
        print("    Copying %s to %s" % (rhdfile, locpath))
        shutil.copy(rhdfile, locpath)
    if len(stimtxt_paths)>0:
        print("    Copying %s to %s" % (stimtxt_path, locpath))
        shutil.copy(stimtxt_path, locpath)

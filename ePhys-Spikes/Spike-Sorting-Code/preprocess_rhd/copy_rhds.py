import shutil
import os
import glob
import warnings

rawfolder = "/media/G1/xl_neurovascular_coupling/November_2021 - NVC/BC8/"
locfolder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/data/BC8"
if not os.path.exists(locfolder):
    os.makedirs(locfolder)


sessions = [
    # "2021-11-18",
    # "2021-11-19",
    # "2021-11-24",
    # "2021-11-30-a",
    # "2021-11-30-b",
    # "2021-12-05-a",
    # "2021-12-05-b",
    # "2021-12-06",
    # "2021-12-13",
    # "2021-12-19",
    # "2021-12-26",
    "2022-01-9",
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

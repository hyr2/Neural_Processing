import shutil
import os

rawfolder = "/media/G1/xl_neurovascular_coupling/November_2021 - NVC/B-BC5/"
locfolder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/data/B-BC5"
if not os.path.exists(locfolder):
    os.makedirs(locfolder)

sessions = [
    # "1-6-2022",
    # "1-8-2022",
    # "1-12-2022",
    # "1-17-2022",
    # "1-24-2022",
    # "1-31-2022"
    "2-26-2022"
]

for session in sessions:
    rawpath = os.path.join(rawfolder, session, "ePhys")
    stimtxtpath = os.path.join(rawfolder, session, "LSCI", "whisker_stim.txt")
    locpath = os.path.join(locfolder, session)
    if not os.path.exists(locpath):
        os.makedirs(locpath)
    rhdfiles = list(filter(lambda x: x.endswith(".rhd"), os.listdir(rawpath)))
    print(rawpath)
    for rhdfile in rhdfiles:
        print("    Copying %s to %s" % (os.path.join(rawpath, rhdfile), locpath))
        shutil.copy(os.path.join(rawpath, rhdfile), locpath)
    print("    Copying %s to %s" % (stimtxtpath, locpath))
    shutil.copy(stimtxtpath, locpath)


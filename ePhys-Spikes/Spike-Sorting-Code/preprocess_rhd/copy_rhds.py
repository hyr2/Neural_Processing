import shutil
import os
​
rawfolder = "/media/G1/xl_neurovascular_coupling/November_2021 - NVC/BC6/"
locfolder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/data/BC6"
if not os.path.exists(locfolder):
    os.makedirs(locfolder)
​
sessions = [
    #"11-11-2021", "11-23-2021",
    #"12-30-2022", "11-18-2021",  "11-30-2021", 
    "12-07-2021", # this date is invalid: main whisker gone.
    # "11-2-2021", 
    # "11-4-2021", 
    # "12-14-2021"
]
​
for session in sessions:
    rawpath = os.path.join(rawfolder, session, "ePhys", "NVC", "data_211207_152726")
    stimtxtpath = os.path.join(rawpath, "whisker_stim.txt")
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

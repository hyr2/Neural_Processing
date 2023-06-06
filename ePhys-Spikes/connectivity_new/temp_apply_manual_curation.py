import os

import numpy as np
import pandas as pd
import config as cfg

manual_curation_list = [
    # animal ID, session id, i_connec, curation action
    ("rh9", "23-01-11", 10, "k-d"),
    ("rh9", "23-02-08", 0, "k-d"),
    ("bc7", "21-12-06", 39, "k-d"),
    ("bc7", "21-12-06", 48, "k-d"),
    ("bc7", "21-12-06", 133, "k-d"),
    ("bc7", "21-12-07", 50, "k-d"),
    ("bc7", "21-12-07", 96, "k-d"),
    ("bc7", "21-12-07", 109, "k-d"),
    ("bc7", "21-12-07", 125, "k-d"),
    ("bc7", "21-12-24", 66, "k-d"),
    ("rh8", "22-12-01", 28, "k-d"),
    ("rh8", "22-12-01", 29, "k-d"),
    ("rh8", "22-12-03", 25, "k-d"),
    ("rh7", "22-12-02", 27, "k-d"),
    ("rh7", "22-12-02", 29, "k-d"),
    ("rh3", "22-10-19", 20, "d-k"),
    ("rh3", "22-10-19", 57, "k-d"),
    ("rh3", "22-11-02", 10, "d-k"),
    ("rh3", "22-11-02", 33, "d-k"),
]
for (animal_id, session_id, i_connec, action) in manual_curation_list:
    csvname = os.path.join(
        cfg.con_resdir, "processed_data_"+animal_id,
        session_id, "conn_keep_mask.csv"
    )
    mask = pd.read_csv(csvname, header=None).values.astype(int).squeeze()
    print(csvname, i_connec)
    change_flag = False
    if action=="k-d":
        if mask[i_connec]==1:
            mask[i_connec] = 0
            change_flag = True
        else:
            print("    Already rejected")
    else:
        if mask[i_connec]==0:
            mask[i_connec] = 1
            change_flag = True
        else:
            print("    Already accepted")
    # write back
    if change_flag:
        print("Writing back")
        pd.DataFrame(data=mask).to_csv(csvname, index=False, header=False)
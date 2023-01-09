import os

import numpy as np
import pandas as pd
from natsort import natsorted

postproc_dir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out/B-BC5"
postproc_subfolders = natsorted(os.listdir(postproc_dir))

for iter, subdirname in enumerate(postproc_subfolders):
    full_dir = os.path.join(postproc_dir, subdirname)
    if os.path.isdir(full_dir):
        x = pd.read_csv(os.path.join(full_dir, "accept_mask.csv"), header=None).values
        print(iter, ' ', subdirname, np.sum(x), '/', x.shape[0])

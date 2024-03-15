# Population coupling script

import numpy as np
import os

input_dir_tmp = '/home/hyr2-office/Documents/Data/NVC/Tracking/processed_data_rh11/Shank_3/Processed/count_analysis'

np.load(os.path.join(input_dir_tmp,'all_clus_property.npy'))
np.load(os.path.join(input_dir_tmp,'all_clus_property_star.npy'))

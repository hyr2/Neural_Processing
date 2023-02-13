# 1 read RHD and save as MDA for accessing arbitrary segment
#     (TODO figure out a way to do downsampling by chunk while keeping the sample alignment)
# 2 get trial stamps and read corresponding segments of ePhys
# 3 downsample, filter, Hilbert.
# 4 cross-correlation with spike

import os, sys
from copy import deepcopy
import gc

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from natsort import natsorted

sys.path.append("../ePhys-Spikes/Spike-Sorting-Code/preprocess_rhd/")
from load_intan_rhd_format import read_data, read_header, get_n_samples_in_data, read_stimtxt
from utils.mdaio import DiskWriteMda
from utils.write_mda import writemda16i
from utils.filtering import notch_filter
from utils_cc import logtofile


#%%
def check_header_consistency(hA, hB):
    if len(hA)!=len(hB): 
        return False
    for a, b in zip(hA, hB):
        if a!=b:
            return False
    return True

def intersection_lists(input_dict):
    # input_dict is a dictionary of lists of native channel orders for each .rhd file in one session. If there are five .rhd files in the session, then the len(input_dict) will be 5.
    # num_files = len(input_dict)
    intersection_list_native_ch_order = list(range(0,256))  # only for 128 channels of the intan
    for i in input_dict:
        intersection_list_native_ch_order = list(set(input_dict[i]) & set(intersection_list_native_ch_order))
    return intersection_list_native_ch_order,len(intersection_list_native_ch_order)

def func_savelfp(input_rootdir, output_rootdir, rel_dir):

    Raw_dir = os.path.join(input_rootdir, rel_dir)
    output_dir = os.path.join(output_rootdir, rel_dir)
    filenames = os.listdir(Raw_dir)
    filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
    filenames = natsorted(filenames)
    
    print(filenames)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # read rhd files and append data in list 
    ### REMEMBER native order starts from 0 ###
    n_samples_cumsum_by_file = [0]
    n_samples = 0
    for filename in filenames:
        # with logtofile():
        n_ch, n_samples_this_file = get_n_samples_in_data(os.path.join(Raw_dir, filename))
        n_samples += n_samples_this_file
        n_samples_cumsum_by_file.append(n_samples) #count of total V points
    
    # Fix implemented: user rejected a channel during the recording/experiment
    dict_ch_nativeorder_allsessions = {}
    for i_file,filename in enumerate(filenames):
        with logtofile():
            with open(os.path.join(Raw_dir, filename), "rb") as fh:
                head_dict = read_header(fh)
        chs_info_local = deepcopy(head_dict['amplifier_channels'])
        chs_native_order_local = [e['native_order'] for e in chs_info_local]
        dict_ch_nativeorder_allsessions[i_file] = chs_native_order_local   
        
    TrueNativeChOrder,n_ch = intersection_lists(dict_ch_nativeorder_allsessions)
    # writer = DiskWriteMda(os.path.join(output_dir, "converted_data.mda"), (n_ch, n_samples), dt="int16")

    for i_file, filename in enumerate(filenames):
        print("----%s----"%(os.path.join(Raw_dir, filename)))
        # data_dict = read_data(os.path.join(Raw_dir, filename))
        # chs_info = deepcopy(data_dict['amplifier_channels'])
        # chs_native_order = [e['native_order'] for e in chs_info]
        # reject_ch_indx = np.where(chs_native_order == np.setdiff1d(chs_native_order,TrueNativeChOrder))[0]
        # # print("Applying notch")
        # print("Data are read") # no need to notch since we only care about 250~5000Hz
        # ephys_data = data_dict['amplifier_data']
        # # ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
        # ephys_data = ephys_data.astype(np.int16)
        # ephys_data = np.delete(ephys_data,reject_ch_indx,axis = 0)
        # print("Appending chunk to disk")
        # entry_offset = n_samples_cumsum_by_file[i_file] * n_ch
        # writer.writeChunk(ephys_data, i1=0, i2=entry_offset)
        # del(ephys_data)
        # del(data_dict)
        # del(reject_ch_indx)
        # gc.collect()


rawdir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/data/"
outputdir = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/mytempfolder/"
rel_dir = "RH-8/2022-12-03"
func_savelfp(rawdir, outputdir, rel_dir)
# This script is used to create geom.csv file for the channel mapping. In addition, it is also used to generate the trial_times.mat

# Some notes:
    # 1. Only for 128 channel recordings. Some functions inherently assume this. Can perform both 2x16 and 1x32 
    # 2. if the file: min_chanmap_mask.npy exists in the session path, it will be applied to create a minimum channel map mask (reflected in geom.csv + converted_data.mda + native_ch_order.npy)

#%%
import os, gc, warnings, json, sys, glob, shutil
from copy import deepcopy
sys.path.append(os.path.join(os.getcwd(),'Intan-File-Reader'))
sys.path.append(os.getcwd())
from load_intan_rhd_format import Support
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from load_intan_rhd_format import read_data, read_header, get_n_samples_in_data, read_stimtxt
from utils.mdaio import DiskWriteMda
from utils.write_mda import writemda16i
from utils.filtering import notch_filter
from natsort import natsorted
from matplotlib import pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000

# Check header consistency
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

def func_preprocess(Raw_dir, output_dir, ELECTRODE_2X16, CHANNEL_MAP_FPATH):

    # channel spacing
    # 32x1 per shank setting
    if not ELECTRODE_2X16:
        GW_BETWEEN_SHANK = 300 # micron
        GH = 25 # micron
    else:
        # 16x2 per shank setting
        GW_BETWEEN_SHANK = 250
        GW_WITHIN_SHANK = 30
        GH = 30
    
    
    #/media/luanlab/DATA/SpikeSorting/RawData/2021-09-04B-aged/2022-01-01/2022-01-01_moving
    
    # given a session
    # Raw_dir  = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/HR/11-2'
    SESSION_REL_PATH = Raw_dir.split('/')[-1]
    # output_dir  = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/Haad/RH3/'
    # Raw_dir = os.path.join(Raw_dir, SESSION_REL_PATH)
    SESSION_FOLDER_CSV = os.path.join(output_dir, SESSION_REL_PATH)
    SESSION_FOLDER_MDA = SESSION_FOLDER_CSV
    
    filename_min_chan_mask = os.path.join(Raw_dir,'min_chanmap_mask_new.npy')
    
    filename_trials_export = os.path.join(SESSION_FOLDER_CSV,'trials_times.mat')
    filename_trials_digIn = os.path.join(SESSION_FOLDER_CSV,'trials_digIn.png')
    source_dir_list = natsorted(os.listdir(Raw_dir))
    matlabTXT = 'whisker_stim.txt'
    matlabTXT = os.path.join(Raw_dir,matlabTXT)
    del source_dir_list
    
    if os.path.isfile(matlabTXT):
        # Read .txt file
        stim_start_time, stim_num, seq_period, len_trials, num_trials, FramePerSeq, total_seq, len_trials_arr = read_stimtxt(matlabTXT)
    
    filenames = os.listdir(Raw_dir)
    filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
    filenames = natsorted(filenames)
    
    print(filenames)
    
    if not os.path.exists(SESSION_FOLDER_MDA):
        os.makedirs(SESSION_FOLDER_MDA)
    
    if not os.path.exists(SESSION_FOLDER_CSV):
        os.makedirs(SESSION_FOLDER_CSV)

    
    # read rhd files and append data in list 
    ### REMEMBER native order starts from 0 ###
    n_samples_cumsum_by_file = [0]
    n_samples = 0
    for filename in filenames:
        n_ch, n_samples_this_file = get_n_samples_in_data(os.path.join(Raw_dir, filename))
        n_samples += n_samples_this_file
        n_samples_cumsum_by_file.append(n_samples) #count of total V points
    
    # Fix implemented: user rejected a channel during the recording/experiment
    dict_ch_nativeorder_allsessions = {}
    for i_file,filename in enumerate(filenames):
        with open(os.path.join(Raw_dir, filename), "rb") as fh:
            head_dict = read_header(fh)
        chs_info_local = deepcopy(head_dict['amplifier_channels'])
        chs_native_order_local = [e['native_order'] for e in chs_info_local]
        dict_ch_nativeorder_allsessions[i_file] = chs_native_order_local   
        
    TrueNativeChOrder,n_ch = intersection_lists(dict_ch_nativeorder_allsessions)
    
    chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
    if np.min(chmap_mat)==1:
        print("    Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
        chmap_mat -= 1
    
    if os.path.isfile(filename_min_chan_mask):
        arr_mask = np.load(filename_min_chan_mask)
        # geom_map = -1*np.ones((len(arr_mask), 2), dtype=np.int)
        ch_id_to_reject = []
        for iter_localA in range(len(TrueNativeChOrder)):
            loc_local = np.squeeze(np.argwhere(chmap_mat == TrueNativeChOrder[iter_localA]))
            if not ELECTRODE_2X16: 
                r_id = loc_local[0]
                sh_id = loc_local[1]
            else:
                r_id = loc_local[0]
                sh_id = loc_local[1] // 2   # floor division gets us the effective shank ID 
            if (arr_mask[r_id,sh_id] == False):
                ch_id_to_reject.append(TrueNativeChOrder[iter_localA])
        indx_to_reject = np.where(np.isin(TrueNativeChOrder,ch_id_to_reject))[0]
        TrueNativeChOrder_new = np.delete(TrueNativeChOrder,indx_to_reject)
        # arr_mask = np.load(filename_min_chan_mask)
        # n_ch = arr_mask.shape[0]
        n_ch = TrueNativeChOrder_new.shape[0]
        
    writer = DiskWriteMda(os.path.join(SESSION_FOLDER_MDA, "converted_data.mda"), (n_ch, n_samples), dt="int16")
        
    chs_impedance = None
    notch_freq = None
    chs_native_order = None
    sample_freq = None
    df_final = pd.DataFrame(columns=['Time','ADC'])
    for i_file, filename in enumerate(filenames):
        print("----%s----"%(os.path.join(Raw_dir, filename)))
        with open(os.path.join(Raw_dir, filename), "rb") as fh:
            head_dict = read_header(fh)
        # Saving sampling rate info for future use
        if i_file == 0:
            dictionary_summary = {
                "Session": filename[0:filename.index('_')],
                "NumChannels": head_dict['num_amplifier_channels'],
                "SampleRate": head_dict['sample_rate'],
                "ELECTRODE_2X16": ELECTRODE_2X16,
                "Notch filter": head_dict['notch_filter_frequency'],
                "SequenceTime":seq_period,
                "StimulationTime":stim_num*seq_period,
                "StimulationStartTime":stim_start_time,
                "SeqPerTrial":len_trials,
                "NumTrials":num_trials,
                "FPS":FramePerSeq
            }
            # Serializing json
            json_object = json.dumps(dictionary_summary, indent=4)
            with open(os.path.join(SESSION_FOLDER_MDA,'pre_MS.json'), "w") as outfile:
                outfile.write(json_object)
    
        data_dict = read_data(os.path.join(Raw_dir, filename))
        chs_info = deepcopy(data_dict['amplifier_channels'])
        
        
        arr_ADC = data_dict['board_dig_in_data']                       # Digital Trigger input 
        Time = data_dict['t_amplifier']                        		# Timing info from INTAN
        if arr_ADC.shape[0] != 1:                                  # For data with multiple digital inputs (dig input channel 1 is the speckle trigger)
            arr_ADC = arr_ADC[0,:]
        arr_ADC = np.reshape(arr_ADC,(arr_ADC.size,))
        df = {'Time':Time,'ADC':arr_ADC}
        df = pd.DataFrame(df,dtype = np.single)
        df_final = pd.concat([df_final,df],axis = 0,ignore_index=True)
        
        # record and check key information
        if chs_native_order is None:
            chs_native_order = [e['native_order'] for e in chs_info]
            chs_impedance = [e['electrode_impedance_magnitude'] for e in chs_info]
            print("    #Chans with >= 3MOhm impedance:", np.sum(np.array(chs_impedance)>=3e6))
            notch_freq = head_dict['notch_filter_frequency']
            sample_freq = head_dict['sample_rate']
        else:
            tmp_native_order = [e['native_order'] for e in chs_info]
            print("    #Chans with >= 3MOhm impedance:", np.sum(np.array([e['electrode_impedance_magnitude'] for e in chs_info])>=3e6))
            if not check_header_consistency(tmp_native_order, chs_native_order):
                warnings.warn("WARNING in preprocess_rhd: native ordering of channels inconsistent within one session\n")
            if notch_freq != head_dict['notch_filter_frequency']:
                warnings.warn("WARNING in preprocess_rhd: notch frequency inconsistent within one session\n")
            if sample_freq != head_dict['sample_rate']:
                warnings.warn("WARNING in preprocess_rhd: sampling frequency inconsistent within one session\n")
        
        chs_native_order = [e['native_order'] for e in chs_info]
        reject_ch_indx = np.where(np.any(chs_native_order==(np.setdiff1d(chs_native_order,TrueNativeChOrder)[:,None]), axis=0))[0]
        # only implement min channel map mask when the file exists 
        # thus this addition will not affect normal spike sorting pipeline
        if os.path.isfile(filename_min_chan_mask): 
            ch_id_to_reject = []
            arr_mask = np.load(filename_min_chan_mask)
            for iter_localA in range(len(chs_native_order)):
                loc_local = np.squeeze(np.argwhere(chmap_mat == chs_native_order[iter_localA]))
                if not ELECTRODE_2X16: 
                    r_id = loc_local[0]
                    sh_id = loc_local[1]
                else:
                    r_id = loc_local[0]
                    sh_id = loc_local[1] // 2   # floor division gets us the effective shank ID 
                if (arr_mask[r_id,sh_id] == False):
                    ch_id_to_reject.append(chs_native_order[iter_localA])
                    
            # arr1 = np.setdiff1d(chs_native_order,arr_mask)
            # arr2 = np.squeeze(chs_native_order)
            # indx_to_reject = np.where(np.isin(arr2,arr1))[0]
            # andd = np.logical_and(chmap_mat ,arr_mask)
            # linear_arr_min = chmap_mat[andd]
            # linear_arr_min = deepcopy(np.sort(linear_arr_min,kind = 'mergesort'))
            
            indx_to_reject = np.where(np.isin(chs_native_order,ch_id_to_reject))[0]
            reject_ch_indx = np.append(reject_ch_indx,indx_to_reject)
        print("    Intan channels to reject:", reject_ch_indx)
        # print("Applying notch")
        print("    Data are read") # no need to notch since we only care about 250~5000Hz
        ephys_data = data_dict['amplifier_data']
        ephys_data = np.delete(ephys_data,reject_ch_indx,axis = 0)
        print("    Notching + CMR of medians")
        if notch_freq < 1:
            print("      RHD says notch_freq=%.2f, we'll do 60."%(notch_freq))
            ephys_data = notch_filter(ephys_data, sample_freq, 60, Q=20)
        else:
            ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
        ephys_data = ephys_data - np.median(ephys_data, axis=0)
        ephys_data = ephys_data.astype(np.int16)
        
        print("    Appending chunk to disk")
        entry_offset = n_samples_cumsum_by_file[i_file] * n_ch
        writer.writeChunk(ephys_data, i1=0, i2=entry_offset)
        del(ephys_data)
        del(data_dict)
        del(df)
        del(reject_ch_indx)
        gc.collect()
    
    # Saving trial_times.mat
    arr_Time = pd.Series(df_final.Time)          # Time in seconds
    arr_Time = arr_Time.to_numpy(dtype = np.single)
    arr_ADC = pd.Series(df_final.ADC)            # ADC input (CMOS trigger)
    arr_ADC = arr_ADC.to_numpy(dtype = np.single)
    arr_ADC[arr_ADC >= 1] = 5                # Ceiling the ADC data (ideal signal)
    arr_ADC[arr_ADC < 1] = 0                # Flooring the ADC data (ideal signal)

    # If data was taken in two sessions:
    # If experiment was taken in two sessions (e.g: a b only) -------------------------------
    temp_arr = (arr_Time - np.roll(arr_Time,-1) > 1)
    temp_arr = np.where(temp_arr)
    iter = temp_arr[0]
    if len(iter) > 1:
        iter = np.delete(iter,[-1])
        iter = iter[0]
        temp_arr = arr_Time[iter+1:]
        temp_arr = temp_arr + arr_Time[iter]
        arr_Time = np.delete(arr_Time,np.arange(iter+1,len(arr_Time),1))
        arr_Time = np.concatenate((arr_Time,temp_arr))

    # Finding peaks
    arr_ADC_diff = np.diff(arr_ADC)
    arr_ADC_diff[arr_ADC_diff<0] = 0
    arr_Time_diff = np.delete(arr_Time,[-1])
    timestamp_frame = ( arr_ADC_diff - np.roll(arr_ADC_diff,1) > 0.5) & (arr_ADC_diff - np.roll(arr_ADC_diff,-1) > 0.5) # for digital
    # Here I compute the indices of the timestamps 
    timestamp_frame = timestamp_frame.nonzero()[0]                                        # Timestamp indices of the frames (FOIL Camera)
    # sequences
    temp_vec = np.diff(timestamp_frame)
    x = np.argwhere(temp_vec > sample_freq*0.03)                                                   # Detect sequences
    x = x.astype(int)
    x = np.reshape(x,(len(x),))
    x+=1
    x = np.insert(x,0,0)                                                                  # So that we dont miss the first seq
    timestamp_seq = timestamp_frame[x]
    # trials
    xx = np.argwhere(temp_vec > sample_freq*1)                                                      # Detect trials
    xx = xx.astype(int)
    xx = np.reshape(xx,(len(xx),))
    xx+=1
    xx = np.insert(xx,0,0)    
    
    # xx = np.delete(xx,-1)   # extra
    
    timestamp_trials = timestamp_frame[xx]
    
    # Actual timestamps of the sequences and trials
    timestamp_seq_times = arr_Time[timestamp_seq]           # in seconds
    timestamp_trials_times = arr_Time[timestamp_trials]     # in seconds
    
    #----------------------------- Plotting -----------------------------(suppressed)
    plt.figure()
    plt.plot(arr_Time,arr_ADC)
    plt.plot(timestamp_seq_times,arr_ADC[timestamp_frame[x]]+1,'ro')
    plt.plot(timestamp_trials_times,arr_ADC[timestamp_frame[xx]]+1,'go')
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches((16, 9), forward=False)
    fig.savefig(filename_trials_digIn, dpi=200, format = 'png')
    
    # Exporting Timestamps of the trial start times:
    tt_export = timestamp_frame[xx]
    export_timestamps_trials = {'empty':[0],'t_trial_start':tt_export}
    savemat(filename_trials_export,export_timestamps_trials)
    
    # generate geom.csv
    
    # read .mat for channel map (make sure channel index starts from 0)
    if not ELECTRODE_2X16:
        print(chmap_mat.shape)
        if chmap_mat.shape!=(32,4):
            raise ValueError("Channel map is of shape %s, expected (32,4)" % (chmap_mat.shape))
            
            
        if os.path.isfile(filename_min_chan_mask):
            arr_mask = np.load(filename_min_chan_mask)
            geom_map = -1*np.ones((len(TrueNativeChOrder_new), 2), dtype= int)
            for i, native_order in enumerate(TrueNativeChOrder_new):
                loc = np.where(chmap_mat==native_order)
                geom_map[i,0] = loc[1][0]*GW_BETWEEN_SHANK
                geom_map[i,1] = loc[0][0]*GH
        else:
            # find correct locations for valid chanels
            geom_map = -1*np.ones((len(TrueNativeChOrder), 2), dtype= int)
            for i, native_order in enumerate(TrueNativeChOrder):
                loc = np.where(chmap_mat==native_order)
                geom_map[i,0] = loc[1][0]*GW_BETWEEN_SHANK
                geom_map[i,1] = loc[0][0]*GH
                
        
    else:
        print(chmap_mat.shape)
        if chmap_mat.shape!=(16,8):
            raise ValueError("Channel map is of shape %s, expected (16,8)" % (chmap_mat.shape))
        # Generating the geom.csv file required by mountainsort       
        if os.path.isfile(filename_min_chan_mask):
            arr_mask = np.load(filename_min_chan_mask)
            geom_map = -1*np.ones((len(TrueNativeChOrder_new), 2), dtype=np.int)
            for i, native_order in enumerate(TrueNativeChOrder_new):
                loc = np.where(chmap_mat==native_order)
                geom_map[i,0] = (loc[1][0]//2)*GW_BETWEEN_SHANK + (loc[1][0]%2)*GW_WITHIN_SHANK
                geom_map[i,1] = loc[0][0]*GH
        else:
            geom_map = -1*np.ones((len(TrueNativeChOrder), 2), dtype=np.int)
            for i, native_order in enumerate(TrueNativeChOrder):
                loc = np.where(chmap_mat==native_order)
                geom_map[i,0] = (loc[1][0]//2)*GW_BETWEEN_SHANK + (loc[1][0]%2)*GW_WITHIN_SHANK
                geom_map[i,1] = loc[0][0]*GH
                
    # generate .csv
    geom_map_df = pd.DataFrame(data=geom_map)
    geom_map_df.to_csv(os.path.join(SESSION_FOLDER_CSV, "geom.csv"), index=False, header=False)
    print("Geom file generated")
    infodict = {"sample_freq": sample_freq}
    if os.path.isfile(matlabTXT):
        try:
            shutil.copy(matlabTXT,SESSION_FOLDER_CSV)
        except shutil.SameFileError:
            pass
    # savemat(os.path.join(SESSION_FOLDER_CSV, "info.mat"), infodict)
    with open(os.path.join(SESSION_FOLDER_CSV, "info.json"), "w") as fjson:
        json.dump(infodict, fjson)
    np.save(os.path.join(SESSION_FOLDER_CSV, "native_ch_order.npy"), TrueNativeChOrder)
    print("    Done!")

def func_preprocess_spont(Raw_dir, output_dir, ELECTRODE_2X16, CHANNEL_MAP_FPATH):

    # channel spacing
    # 32x1 per shank setting
    if not ELECTRODE_2X16:
        GW_BETWEEN_SHANK = 300 # micron
        GH = 25 # micron
    else:
        # 16x2 per shank setting, MirroHippo Fei Old Device ChMap
        GW_BETWEEN_SHANK = 250
        GW_WITHIN_SHANK = 30
        GH = 30
    
    
    #/media/luanlab/DATA/SpikeSorting/RawData/2021-09-04B-aged/2022-01-01/2022-01-01_moving
    
    # given a session
    # Raw_dir  = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/data/HR/11-2'
    SESSION_REL_PATH = Raw_dir.split('/')[-1]
    # output_dir  = '/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/Haad/RH3/'
    # Raw_dir = os.path.join(Raw_dir, SESSION_REL_PATH)
    SESSION_FOLDER_CSV = os.path.join(output_dir, SESSION_REL_PATH)
    SESSION_FOLDER_MDA = SESSION_FOLDER_CSV
    
    filename_min_chan_mask = os.path.join(Raw_dir,'min_chanmap_mask_new.npy')
    
    filename_trials_export = os.path.join(SESSION_FOLDER_CSV,'trials_times.mat')
    filename_trials_digIn = os.path.join(SESSION_FOLDER_CSV,'trials_digIn.png')
    source_dir_list = natsorted(os.listdir(Raw_dir))
    del source_dir_list

    
    filenames = os.listdir(Raw_dir)
    filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
    filenames = natsorted(filenames)
    
    print(filenames)
    
    if not os.path.exists(SESSION_FOLDER_MDA):
        os.makedirs(SESSION_FOLDER_MDA)
    
    if not os.path.exists(SESSION_FOLDER_CSV):
        os.makedirs(SESSION_FOLDER_CSV)

    
    # read rhd files and append data in list 
    ### REMEMBER native order starts from 0 ###
    n_samples_cumsum_by_file = [0]
    n_samples = 0
    for filename in filenames:
        n_ch, n_samples_this_file = get_n_samples_in_data(os.path.join(Raw_dir, filename))
        n_samples += n_samples_this_file
        n_samples_cumsum_by_file.append(n_samples) #count of total V points
    
    # Fix implemented: user rejected a channel during the recording/experiment
    dict_ch_nativeorder_allsessions = {}
    for i_file,filename in enumerate(filenames):
        with open(os.path.join(Raw_dir, filename), "rb") as fh:
            head_dict = read_header(fh)
        chs_info_local = deepcopy(head_dict['amplifier_channels'])
        chs_native_order_local = [e['native_order'] for e in chs_info_local]
        dict_ch_nativeorder_allsessions[i_file] = chs_native_order_local   
        
    TrueNativeChOrder,n_ch = intersection_lists(dict_ch_nativeorder_allsessions)
    
    chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
    if np.min(chmap_mat)==1:
        print("    Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
        chmap_mat -= 1
    
    if os.path.isfile(filename_min_chan_mask):
        arr_mask = np.load(filename_min_chan_mask)
        # geom_map = -1*np.ones((len(arr_mask), 2), dtype=np.int)
        ch_id_to_reject = []
        for iter_localA in range(len(TrueNativeChOrder)):
            loc_local = np.squeeze(np.argwhere(chmap_mat == TrueNativeChOrder[iter_localA]))
            if not ELECTRODE_2X16: 
                r_id = loc_local[0]
                sh_id = loc_local[1]
            else:
                r_id = loc_local[0]
                sh_id = loc_local[1] // 2   # floor division gets us the effective shank ID 
            if (arr_mask[r_id,sh_id] == False):
                ch_id_to_reject.append(TrueNativeChOrder[iter_localA])
        indx_to_reject = np.where(np.isin(TrueNativeChOrder,ch_id_to_reject))[0]
        TrueNativeChOrder_new = np.delete(TrueNativeChOrder,indx_to_reject)
        # arr_mask = np.load(filename_min_chan_mask)
        # n_ch = arr_mask.shape[0]
        n_ch = TrueNativeChOrder_new.shape[0]
        
    writer = DiskWriteMda(os.path.join(SESSION_FOLDER_MDA, "converted_data.mda"), (n_ch, n_samples), dt="int16")
        
    chs_impedance = None
    notch_freq = None
    chs_native_order = None
    sample_freq = None
    df_final = pd.DataFrame(columns=['Time','ADC'])
    for i_file, filename in enumerate(filenames):
        print("----%s----"%(os.path.join(Raw_dir, filename)))
        with open(os.path.join(Raw_dir, filename), "rb") as fh:
            head_dict = read_header(fh)
        # Saving sampling rate info for future use
        if i_file == 0:
            dictionary_summary = {
                "Session": filename[0:filename.index('_')],
                "NumChannels": head_dict['num_amplifier_channels'],
                "SampleRate": head_dict['sample_rate'],
                "ELECTRODE_2X16": ELECTRODE_2X16,
                "Notch filter": head_dict['notch_filter_frequency'],
            }
            # Serializing json
            json_object = json.dumps(dictionary_summary, indent=4)
            with open(os.path.join(SESSION_FOLDER_MDA,'pre_MS.json'), "w") as outfile:
                outfile.write(json_object)
    
        data_dict = read_data(os.path.join(Raw_dir, filename))
        chs_info = deepcopy(data_dict['amplifier_channels'])
        
        print("Spont. ephys data only")
        
        # record and check key information
        if chs_native_order is None:
            chs_native_order = [e['native_order'] for e in chs_info]
            chs_impedance = [e['electrode_impedance_magnitude'] for e in chs_info]
            print("    #Chans with >= 3MOhm impedance:", np.sum(np.array(chs_impedance)>=3e6))
            notch_freq = head_dict['notch_filter_frequency']
            sample_freq = head_dict['sample_rate']
        else:
            tmp_native_order = [e['native_order'] for e in chs_info]
            print("    #Chans with >= 3MOhm impedance:", np.sum(np.array([e['electrode_impedance_magnitude'] for e in chs_info])>=3e6))
            if not check_header_consistency(tmp_native_order, chs_native_order):
                warnings.warn("WARNING in preprocess_rhd: native ordering of channels inconsistent within one session\n")
            if notch_freq != head_dict['notch_filter_frequency']:
                warnings.warn("WARNING in preprocess_rhd: notch frequency inconsistent within one session\n")
            if sample_freq != head_dict['sample_rate']:
                warnings.warn("WARNING in preprocess_rhd: sampling frequency inconsistent within one session\n")
        
        chs_native_order = [e['native_order'] for e in chs_info]
        reject_ch_indx = np.where(np.any(chs_native_order==(np.setdiff1d(chs_native_order,TrueNativeChOrder)[:,None]), axis=0))[0]
        # only implement min channel map mask when the file exists 
        # thus this addition will not affect normal spike sorting pipeline
        if os.path.isfile(filename_min_chan_mask): 
            ch_id_to_reject = []
            arr_mask = np.load(filename_min_chan_mask)
            for iter_localA in range(len(chs_native_order)):
                loc_local = np.squeeze(np.argwhere(chmap_mat == chs_native_order[iter_localA]))
                if not ELECTRODE_2X16: 
                    r_id = loc_local[0]
                    sh_id = loc_local[1]
                else:
                    r_id = loc_local[0]
                    sh_id = loc_local[1] // 2   # floor division gets us the effective shank ID 
                if (arr_mask[r_id,sh_id] == False):
                    ch_id_to_reject.append(chs_native_order[iter_localA])
                    
            # arr1 = np.setdiff1d(chs_native_order,arr_mask)
            # arr2 = np.squeeze(chs_native_order)
            # indx_to_reject = np.where(np.isin(arr2,arr1))[0]
            # andd = np.logical_and(chmap_mat ,arr_mask)
            # linear_arr_min = chmap_mat[andd]
            # linear_arr_min = deepcopy(np.sort(linear_arr_min,kind = 'mergesort'))
            
            indx_to_reject = np.where(np.isin(chs_native_order,ch_id_to_reject))[0]
            reject_ch_indx = np.append(reject_ch_indx,indx_to_reject)
        print("    Intan channels to reject:", reject_ch_indx)
        # print("Applying notch")
        print("    Data are read") # no need to notch since we only care about 250~5000Hz
        ephys_data = data_dict['amplifier_data']
        ephys_data = np.delete(ephys_data,reject_ch_indx,axis = 0)
        print("    Notching + CMR of medians")
        if notch_freq < 1:
            print("      RHD says notch_freq=%.2f, we'll do 60."%(notch_freq))
            ephys_data = notch_filter(ephys_data, sample_freq, 60, Q=20)
        else:
            ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
        ephys_data = ephys_data - np.median(ephys_data, axis=0)
        ephys_data = ephys_data.astype(np.int16)
        
        print("    Appending chunk to disk")
        entry_offset = n_samples_cumsum_by_file[i_file] * n_ch
        writer.writeChunk(ephys_data, i1=0, i2=entry_offset)
        del(ephys_data)
        del(data_dict)
        del(reject_ch_indx)
        gc.collect()
    
    # generate geom.csv
    
    # read .mat for channel map (make sure channel index starts from 0)
    if not ELECTRODE_2X16:
        print(chmap_mat.shape)
        if chmap_mat.shape!=(32,4):
            raise ValueError("Channel map is of shape %s, expected (32,4)" % (chmap_mat.shape))
            
            
        if os.path.isfile(filename_min_chan_mask):
            arr_mask = np.load(filename_min_chan_mask)
            geom_map = -1*np.ones((len(TrueNativeChOrder_new), 2), dtype= int)
            for i, native_order in enumerate(TrueNativeChOrder_new):
                loc = np.where(chmap_mat==native_order)
                geom_map[i,0] = loc[1][0]*GW_BETWEEN_SHANK
                geom_map[i,1] = loc[0][0]*GH
        else:
            # find correct locations for valid chanels
            geom_map = -1*np.ones((len(TrueNativeChOrder), 2), dtype= int)
            for i, native_order in enumerate(TrueNativeChOrder):
                loc = np.where(chmap_mat==native_order)
                geom_map[i,0] = loc[1][0]*GW_BETWEEN_SHANK
                geom_map[i,1] = loc[0][0]*GH
                
        
    else:
        print(chmap_mat.shape)
        if chmap_mat.shape!=(16,8):
            raise ValueError("Channel map is of shape %s, expected (16,8)" % (chmap_mat.shape))
        # Generating the geom.csv file required by mountainsort       
        if os.path.isfile(filename_min_chan_mask):
            arr_mask = np.load(filename_min_chan_mask)
            geom_map = -1*np.ones((len(TrueNativeChOrder_new), 2), dtype=np.int)
            for i, native_order in enumerate(TrueNativeChOrder_new):
                loc = np.where(chmap_mat==native_order)
                geom_map[i,0] = (loc[1][0]//2)*GW_BETWEEN_SHANK + (loc[1][0]%2)*GW_WITHIN_SHANK
                geom_map[i,1] = loc[0][0]*GH
        else:
            geom_map = -1*np.ones((len(TrueNativeChOrder), 2), dtype=np.int)
            for i, native_order in enumerate(TrueNativeChOrder):
                loc = np.where(chmap_mat==native_order)
                geom_map[i,0] = (loc[1][0]//2)*GW_BETWEEN_SHANK + (loc[1][0]%2)*GW_WITHIN_SHANK
                geom_map[i,1] = loc[0][0]*GH
                
    # generate .csv
    geom_map_df = pd.DataFrame(data=geom_map)
    geom_map_df.to_csv(os.path.join(SESSION_FOLDER_CSV, "geom.csv"), index=False, header=False)
    print("Geom file generated")
    infodict = {"sample_freq": sample_freq}
    # savemat(os.path.join(SESSION_FOLDER_CSV, "info.mat"), infodict)
    with open(os.path.join(SESSION_FOLDER_CSV, "info.json"), "w") as fjson:
        json.dump(infodict, fjson)
    np.save(os.path.join(SESSION_FOLDER_CSV, "native_ch_order.npy"), TrueNativeChOrder)
    print("    Done!")


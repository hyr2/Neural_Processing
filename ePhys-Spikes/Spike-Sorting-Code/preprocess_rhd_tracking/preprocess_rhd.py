# script is used for :

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
    
    source_dir_list_upper = natsorted(os.listdir(Raw_dir))
    filename_shank_info = os.path.join(Raw_dir,'shanks_.json')
    
    with open(filename_shank_info, "r") as f:
        shank_info = json.load(f)
    keys = list(shank_info.keys())
    shank_info = list(shank_info.values())  # select these shanks for sorting only
    
    for iter_l in range(len(shank_info)):
        if shank_info[iter_l]:
            local_outputdir =os.path.join(output_dir,f'Shank_{iter_l}')
            if not os.path.exists(local_outputdir):
                os.makedirs(local_outputdir)
    
    filename_trials_export = os.path.join(output_dir,'trials_times.mat')
    filename_trials_digIn = os.path.join(output_dir,'trials_digIn.png')
    filename_min_chan_mask = os.path.join(Raw_dir,'min_chanmap_mask_new.npy')
    
    # Keeping only directories 
    source_dir_list_upper_new = []
    for iter_local,filename in enumerate(source_dir_list_upper):
        if os.path.isdir(os.path.join(Raw_dir,filename)):
            source_dir_list_upper_new.append(filename)
    del source_dir_list_upper
    source_dir_list_upper = source_dir_list_upper_new

    list_session = []
    list_rhd = []
    


    # Storing all file paths to be processed
    for iter_local, filename_u in enumerate(source_dir_list_upper):
        session_rel_path = os.path.join(Raw_dir,filename_u)
        source_dir_list = os.listdir(session_rel_path)
        source_dir_list = list(filter(lambda x: x.endswith(".rhd"), source_dir_list))
        source_dir_list = natsorted(source_dir_list)
        list_rhd.append(source_dir_list)
        list_session.append(filename_u)
    #for iter in range(2):
    #print('\n'.join(list_rhd[0]))
    #print('\n'.join(list_session))

    # Reading channel map
    chmap_mat = loadmat(CHANNEL_MAP_FPATH)['Ch_Map_new']
    if np.min(chmap_mat)==1:
        print("    Subtracted one from channel map to make sure channel index starts from 0 (Original map file NOT changed)")
        chmap_mat -= 1
    # sum(len(sublist) for sublist in list_rhd)

    N_samples_ = 0
    N_samples_cumsum_by_file_ = [0]
    dict_ch_nativeorder_allsessions = {}
    file_i = -1
    # This double loop is for getting the number of samples in each .rhd file
    for iter_local, filename_u in enumerate(list_session):              # Session loop        
        for iter_local_i, filename_i in enumerate(list_rhd[iter_local]):    # RHD loop
            file_i += 1
            filename_rhd = os.path.join(Raw_dir,filename_u,filename_i)
            n_ch, n_samples_this_file = get_n_samples_in_data(filename_rhd)
            N_samples_ += n_samples_this_file
            N_samples_cumsum_by_file_.append(N_samples_) #cumalative count of total V points

            # This block of code also caters for inter-session changes in channels and will 
            # select the common electrodes among all the sessions
            with open(filename_rhd, "rb") as fh:    
                head_dict = read_header(fh)
            chs_info_local = deepcopy(head_dict['amplifier_channels'])
            chs_native_order_local = [e['native_order'] for e in chs_info_local]
            dict_ch_nativeorder_allsessions[file_i] = chs_native_order_local   

    TrueNativeChOrder,n_ch = intersection_lists(dict_ch_nativeorder_allsessions)    # minimum common channels among all sessions
    
    # Number of channels on each shank on the minimum channel map (between all sessions)
    shank_list_local = []
    for iter_localA in range(len(TrueNativeChOrder)):
        loc_local = np.squeeze(np.argwhere(chmap_mat == TrueNativeChOrder[iter_localA]))
        if not ELECTRODE_2X16: 
            r_id = loc_local[0]
            sh_id = loc_local[1]
        else:
            r_id = loc_local[0]
            sh_id = loc_local[1] // 2   # floor division gets us the effective shank ID
        shank_list_local.append(sh_id)
    shank_list_local = np.array(shank_list_local,dtype=np.int8)
    
    
    print(f'Number of samples (total):\t{N_samples_}')
    print('------------------\n-----------------\n----------------\n')
    print(f'Number of channels\t:{n_ch}')
    dict_writers_mda = {}   # dictionary of variables and writer objects for .mda files
    for iter_l in range(len(shank_info)):           # Creating .mda files
        n_ch_this_shank = (shank_list_local == iter_l).sum()
        if shank_info[iter_l]:
            local_outputdir = os.path.join(output_dir,f'Shank_{iter_l}')
            dict_writers_mda["writer{}".format(iter_l)] = DiskWriteMda(os.path.join(local_outputdir, "converted_data.mda"), (n_ch_this_shank, N_samples_), dt="int16")
            # writer = DiskWriteMda(os.path.join(local_outputdir, "converted_data.mda"), (n_ch, N_samples_), dt="int16")

    chs_impedance = None
    notch_freq = None
    chs_native_order = None
    sample_freq = None
    file_i = -1
    df_final = pd.DataFrame(columns=['Time','ADC'])
    # Main loop for generating converted_data.mda file 
    for iter_local, filename_u in enumerate(list_session):              # Session loop        
        for iter_local_i, filename_i in enumerate(list_rhd[iter_local]):    # RHD loop
            # Read all .rhds and write chunks 
            file_i += 1
            filename_rhd = os.path.join(Raw_dir,filename_u,filename_i)
            data_dict = read_data(filename_rhd)
            chs_info = deepcopy(data_dict['amplifier_channels'])
            arr_ADC = data_dict['board_dig_in_data']                    # Digital Trigger input 
            Time = data_dict['t_amplifier']                        		# Timing info from INTAN
            if arr_ADC.shape[0] != 1:                                   # For data with multiple digital inputs (dig input channel 1 is the speckle trigger)
                arr_ADC = arr_ADC[0,:]
            arr_ADC = np.reshape(arr_ADC,(arr_ADC.size,))
            df = {'Time':Time,'ADC':arr_ADC}
            df = pd.DataFrame(df,dtype = np.single)
            df_final = pd.concat([df_final,df],axis = 0,ignore_index=True)
            # record and check key information
            if chs_native_order is None:
                chs_native_order = [e['native_order'] for e in chs_info]
                chs_impedance = [e['electrode_impedance_magnitude'] for e in chs_info]
                print("#Chans with >= 3MOhm impedance:", np.sum(np.array(chs_impedance)>=3e6))
                notch_freq = head_dict['notch_filter_frequency']
                sample_freq = head_dict['sample_rate']
            else:
                tmp_native_order = [e['native_order'] for e in chs_info]
                print("#Chans with >= 3MOhm impedance:", np.sum(np.array([e['electrode_impedance_magnitude'] for e in chs_info])>=3e6))
                if not check_header_consistency(tmp_native_order, chs_native_order):
                    warnings.warn("WARNING in preprocess_rhd: native ordering of channels inconsistent within one session\n")
                if notch_freq != head_dict['notch_filter_frequency']:
                    warnings.warn("WARNING in preprocess_rhd: notch frequency inconsistent within one session\n")
                if sample_freq != head_dict['sample_rate']:
                    warnings.warn("WARNING in preprocess_rhd: sampling frequency inconsistent within one session\n")
            chs_native_order = [e['native_order'] for e in chs_info]
            shank_list_local = []
            for iter_localA in range(len(chs_native_order)):
                loc_local = np.squeeze(np.argwhere(chmap_mat == chs_native_order[iter_localA]))
                if not ELECTRODE_2X16: 
                    r_id = loc_local[0]
                    sh_id = loc_local[1]
                else:
                    r_id = loc_local[0]
                    sh_id = loc_local[1] // 2   # floor division gets us the effective shank ID
                shank_list_local.append(sh_id)
            shank_list_local = np.array(shank_list_local,dtype=np.int8)
            np_chs_native_order = np.array(chs_native_order,dtype=np.int8)
            # reject_ch_indx_global = np.where(np.any(chs_native_order==(np.setdiff1d(chs_native_order,np.array(TrueNativeChOrder))[:,None]), axis=0))[0]
            reject_ch_indx_global = np.squeeze(np.argwhere(np.in1d(chs_native_order,np.setdiff1d(chs_native_order,np.array(TrueNativeChOrder)))))
            
            # divide ephys array by shank here
            for iter_local_local in range(len(shank_info)):
                if shank_info[iter_local_local]:
                    local_NativeChOrder_shank0 = np_chs_native_order[shank_list_local == iter_local_local]   # channels in the current recording located on shankA
                    reject_ch_indx_local = np.squeeze(np.argwhere(np.in1d(chs_native_order,np.setdiff1d(chs_native_order,np.array(local_NativeChOrder_shank0)))))     
                    if np.size(reject_ch_indx_global) != 0:
                        reject_ch_indx_local = np.concatenate((reject_ch_indx_local,reject_ch_indx_global))    # adding globally rejected (due to missing channels)
                    reject_ch_indx_local = np.unique(reject_ch_indx_local)
                    print("Intan channels to reject:", reject_ch_indx_local)    
                    
                    ephys_data = data_dict['amplifier_data']
                    ephys_data = np.delete(ephys_data,reject_ch_indx_local,axis = 0)
                    ephys_data = notch_filter(ephys_data, sample_freq, 60, Q=20)     # Notch filter 60Hz
                    ephys_data = ephys_data - np.median(ephys_data, axis=0)          # CMR: Common Mode rejection
                    ephys_data = ephys_data.astype(np.int16)
                    
                    print("    Appending chunk to disk")
                    entry_offset = N_samples_cumsum_by_file_[file_i] * n_ch
                    dict_writers_mda[f'writer{iter_local_local}'].writeChunk(ephys_data, i1=0, i2=entry_offset)     # write chunk to shank's .mda file
                
            del(ephys_data)
            del(data_dict)
            del(df)
            del(reject_ch_indx_local)
            del(reject_ch_indx_global)
            gc.collect()
    
    matlabTXT = os.path.join(Raw_dir,filename_u,'whisker_stim.txt')
    trial_mask = os.path.join(Raw_dir,filename_u,'trial_mask.csv')
    try:
        stim_start_time, stim_num, seq_period, len_trials, num_trials, FramePerSeq, total_seq, len_trials_arr = read_stimtxt(matlabTXT)
        trial_mask = pd.read_csv(trial_mask, header=None, index_col=False,dtype = bool)
        TRIAL_KEEP_MASK = trial_mask.to_numpy(dtype = bool)
        TRIAL_KEEP_MASK = np.squeeze(TRIAL_KEEP_MASK)
    except ValueError as e:
        print(e)
        
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
        
            # # only implement min channel map mask when the file exists 
            # # thus this addition will not affect normal spike sorting pipeline
            # if os.path.isfile(filename_min_chan_mask): 
            #     ch_id_to_reject = []
            #     arr_mask = np.load(filename_min_chan_mask)
            #     for iter_localA in range(len(chs_native_order)):
            #         loc_local = np.squeeze(np.argwhere(chmap_mat == chs_native_order[iter_localA]))
            #         if not ELECTRODE_2X16: 
            #             r_id = loc_local[0]
            #             sh_id = loc_local[1]
            #         else:
            #             r_id = loc_local[0]
            #             sh_id = loc_local[1] // 2   # floor division gets us the effective shank ID 
            #         if (arr_mask[r_id,sh_id] == False):
            #             ch_id_to_reject.append(chs_native_order[iter_localA])
                        
            #     # arr1 = np.setdiff1d(chs_native_order,arr_mask)
            #     # arr2 = np.squeeze(chs_native_order)
            #     # indx_to_reject = np.where(np.isin(arr2,arr1))[0]
            #     # andd = np.logical_and(chmap_mat ,arr_mask)
            #     # linear_arr_min = chmap_mat[andd]
            #     # linear_arr_min = deepcopy(np.sort(linear_arr_min,kind = 'mergesort'))
                
            #     indx_to_reject = np.where(np.isin(chs_native_order,ch_id_to_reject))[0]
            #     reject_ch_indx = np.append(reject_ch_indx,indx_to_reject)

    
        # Fix implemented: user rejected a channel during the recording/experiment 
        # This block of code also caters for inter-session changes in channels and will 
        # select the common electrodes among all the sessions
        # for i_file,filename in enumerate(source_dir_list):
        #     with open(os.path.join(session_rel_path, filename), "rb") as fh:
        #         head_dict = read_header(fh)
        #     chs_info_local = deepcopy(head_dict['amplifier_channels'])
        #     chs_native_order_local = [e['native_order'] for e in chs_info_local]
        #     dict_ch_nativeorder_allsessions[i_file] = chs_native_order_local   
        # TrueNativeChOrder,n_ch = intersection_lists(dict_ch_nativeorder_allsessions)
    
        
        

        
        # if os.path.isfile(filename_min_chan_mask):
        #     arr_mask = np.load(filename_min_chan_mask)
        #     # geom_map = -1*np.ones((len(arr_mask), 2), dtype=np.int)
        #     ch_id_to_reject = []
        #     for iter_localA in range(len(TrueNativeChOrder)):
        #         loc_local = np.squeeze(np.argwhere(chmap_mat == TrueNativeChOrder[iter_localA]))
        #         if not ELECTRODE_2X16: 
        #             r_id = loc_local[0]
        #             sh_id = loc_local[1]
        #         else:
        #             r_id = loc_local[0]
        #             sh_id = loc_local[1] // 2   # floor division gets us the effective shank ID 
        #         if (arr_mask[r_id,sh_id] == False):
        #             ch_id_to_reject.append(TrueNativeChOrder[iter_localA])
        #     indx_to_reject = np.where(np.isin(TrueNativeChOrder,ch_id_to_reject))[0]
        #     TrueNativeChOrder_new = np.delete(TrueNativeChOrder,indx_to_reject)
        #     # arr_mask = np.load(filename_min_chan_mask)
        #     # n_ch = arr_mask.shape[0]
        #     n_ch = TrueNativeChOrder_new.shape[0]
            
            
        # writer = DiskWriteMda(os.path.join(SESSION_FOLDER_MDA, "converted_data.mda"), (n_ch, n_samples), dt="int16")
        

        
'''
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
'''

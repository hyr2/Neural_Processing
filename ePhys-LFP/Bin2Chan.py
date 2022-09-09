# -*- coding: utf-8 -*-
"""
Created on Sun May 16 20:56:51 2021

@author: Haad-Rathore
"""
import sys, os
sys.path.append(os.path.join(os.getcwd(),'Intan-File-Reader'))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import shutil
from natsort import natsorted
from load_intan_rhd_format import read_data
from Support import append_df_to_excel, read_stimtxt

# Source directory
# |
# |__data*.rhd
# |__data*.rhd
# |__    .
# |__    .
# |__    .
# |__whisker_stim.txt
# |__chan_map_1x32_128ch.xlsx

def Bin2Chan_main(source_dir, CMR):
    
    # Files and folders
    print('Warning: Make sure "whisker_stim.txt" and "chan_map_1x32_128ch.xlsx" files are present in the current directory? \n')
    # source_dir = input('Enter the source directory here: ')
    # CMR = input('Do you want to subtract the median across electrodes? [1/0]\n')
    CMR = int(CMR)
    source_dir_list = natsorted(os.listdir(source_dir))
    output_dir = os.path.join(source_dir,'FullDataSet')
    Raw_dir = os.path.join(source_dir,'Raw')
    os.mkdir(output_dir)
    os.mkdir(Raw_dir)
    
    for file in source_dir_list:
        if file.endswith('.rhd'):
            shutil.move(os.path.join(source_dir,file),os.path.join(Raw_dir,file))
    
    matlabTXT = source_dir_list[source_dir_list.index('whisker_stim.txt')]
    matlabTXT = os.path.join(source_dir,matlabTXT)
    
    # Read .txt file
    stim_start_time, stim_num, seq_period, len_trials, num_trials, FramePerSeq, total_seq, len_trials_arr = read_stimtxt(matlabTXT)
    
    # Reading first file
    Raw_dir_list = natsorted(os.listdir(Raw_dir))
    filename = os.path.join(Raw_dir, Raw_dir_list[0])
    result = read_data(filename)
    
    
    
    # Saving general info
    Num_chan = len(result['amplifier_channels'])
    Fs = result['frequency_parameters']['board_adc_sample_rate']    
    data = {'Num Channels':[Num_chan],'Notch filter':[result['frequency_parameters']['notch_filter_frequency']],'Fs':[Fs]}
    df = pd.DataFrame(data,index=['1'],dtype=np.int32)
    filename_summary = os.path.join(source_dir,'exp_summary.xlsx')
    df.to_excel(filename_summary, index = False)
    data_summary = {'Sequence Time(s)':[seq_period],'Stimulation Time(s)':\
                    [stim_num*seq_period],'Stimulation Start Time(s)':\
                        [stim_start_time],'Seq/trial':[len_trials],'# Trials':[num_trials],'FPS':[FramePerSeq]}
    df = pd.DataFrame(data_summary)
    append_df_to_excel(filename_summary,df, sheet_name='Sheet1', startrow=2, index = False)          # appends into excel files conveniently
    
    filename_chanlist = os.path.join(source_dir,'chan_list.xlsx')    # generating channel index (some channels were set to off during experiment)
    data_chanlist = np.zeros((Num_chan,), dtype = np.int16)
    impedence_list = np.empty([Num_chan,])
    for iter in range(Num_chan):       
        data_chanlist[iter] = result['amplifier_channels'][iter]['native_order']
        impedence_list[iter] = result['amplifier_channels'][iter]['electrode_impedance_magnitude']
    data = {'Intan_index':data_chanlist, 'Impedence':impedence_list}
    df_chanlist = pd.DataFrame(data,dtype = np.int16)
    chan_list = np.asarray(df_chanlist)
    chan_list = np.reshape(chan_list,(Num_chan,2))           # List of channels recorded
    df_chanlist.to_excel(filename_chanlist,index = False, header = True)
    
    # Writing timing and ADC data (first file)
    filename_csv_summary = 'data-timing.csv'
    filename_csv_summary = os.path.join(output_dir,filename_csv_summary)
    # arr_ADC = result['board_adc_data']                          # Analog Trigger input from the CMOS 
    arr_ADC = result['board_dig_in_data']                       # Digital Trigger input 
    Time = result['t_amplifier']                                # Timing info from INTAN
    arr_ADC = np.reshape(arr_ADC,(arr_ADC.size,))
    df = {'Time(s)':Time,'ADC':arr_ADC}
    df = pd.DataFrame(df,dtype = np.single)
    df.to_csv(filename_csv_summary, mode = 'w', header = True, index = False)
    # Writing amplifier data for each channel (first file)
    EEG = result['amplifier_data']
    Median_data_EEG = np.median(EEG, axis = 0)
    for iter_chan in range(Num_chan):
        filename_csv = 'Chan' + str(np.int(chan_list[iter_chan][0])) + '.csv'
        filename_csv = os.path.join(output_dir,filename_csv)
        # EEG = result['amplifier_data'][iter_chan,:]
        if CMR == 1:
            EEG_final = np.subtract(EEG[iter_chan,:],Median_data_EEG)
        else:
            EEG_final = EEG[iter_chan,:]
        df = {'EEG':EEG_final}
        df = pd.DataFrame(df,dtype = np.single)
        df.to_csv(filename_csv, mode = 'w', header = True, index = False)
    
    del Raw_dir_list[0]
    # Looping over the other files
    for filename in Raw_dir_list:
        # Reading file
        filename = os.path.join(Raw_dir, filename)
        result = read_data(filename)
        # Writing timing and arr_ADC data
        # arr_ADC = result['board_adc_data']                      # Analog Trigger input from the CMOS 
        arr_ADC = result['board_dig_in_data']                       # Digital Trigger input 
        Time = result['t_amplifier']                        # Timing info from INTAN
        arr_ADC = np.reshape(arr_ADC,(arr_ADC.size,))
        df = {'Time(s)':Time,'ADC':arr_ADC}
        df = pd.DataFrame(df,dtype = np.single)
        df.to_csv(filename_csv_summary, mode = 'a', header = False, index = False)
        # Writing amplifier data for each channel
        EEG = result['amplifier_data']
        Median_data_EEG = np.median(EEG, axis = 0)
        for iter_chan in range(Num_chan):
            filename_csv = 'Chan' + str(np.int(chan_list[iter_chan][0])) + '.csv'
            filename_csv = os.path.join(output_dir,filename_csv)
            # EEG = result['amplifier_data'][iter_chan,:]     # Electrical recording (extracellular)
            if CMR == 1:
                EEG_final = np.subtract(EEG[iter_chan,:],Median_data_EEG)
            else:
                EEG_final = EEG[iter_chan,:]
            df = {'EEG':EEG_final}
            df = pd.DataFrame(df,dtype = np.single)
            df.to_csv(filename_csv, mode = 'a', header = False, index = False)
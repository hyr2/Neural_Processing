import sys, os, shutil
import numpy as np
import pandas as pd
from natsort import natsorted
from matplotlib import pyplot as plt 
from scipy.io import savemat      # Import function to read data.
sys.path.append(os.getcwd())
from Support import append_df_to_excel

# Source directory
# |
# |__Raw\
# |__FullDataSet\
# |__chan_list.xlsx
# |__exp_summary.xlsx
# |__whisker_stim.txt
# |__chan_map_1x32_128ch.xlsx


def Bin2Trials_main(source_dir):
    
    # Files and folders
    # source_dir = input('Enter the directory source directory here: ')
    Raw_dir = os.path.join(source_dir,'Raw')
    Data_dir = os.path.join(source_dir,'FullDataSet')
    # output_dir = source_dir[:len(source_dir)-11]
    output_dir = os.path.join(source_dir,'Bin2Trials')
    Raw_dir_list = natsorted(os.listdir(Raw_dir))
    Data_dir_list = natsorted(os.listdir(Data_dir))
    source_dir_list = natsorted(os.listdir(source_dir))
    filename_trials_export = os.path.join(source_dir,'trials_times.mat')
    
    os.mkdir(output_dir)
    
    
    # Importing Summary of Optical Data
    exp_summary_dir =  os.path.join(source_dir, source_dir_list[source_dir_list.index('exp_summary.xlsx')])
    df_exp_summary = pd.read_excel(exp_summary_dir)
    arr_exp_summary = df_exp_summary.to_numpy()
    Num_chan = arr_exp_summary[0,0]         # Number of channels
    Notch_freq = arr_exp_summary[0,1]       # Notch frequencey selected (in Hz)
    Fs = arr_exp_summary[0,2]               # Sampling freq (in Hz)
    stim_start_t = arr_exp_summary[2,2]     # stimulation start time
    stim_end_t = stim_start_t + arr_exp_summary[2,1] - 0.05   # Stimulation end time
    len_trial = arr_exp_summary[2,3]        # Number of sequences in a single trial
    num_trials = arr_exp_summary[2,4]        # Number of trials
    FramesPerSeq = arr_exp_summary[2,5]     # Frames per sequence
    len_seq = arr_exp_summary[2,0]          # Time period of one sequence
    
    skip_opticalDelay = 0.0                  # in seconds (delay present while starting optical system by human operator)
    skip_opticalDelay = int(skip_opticalDelay * Fs)              # Number of inital samples to skip  (delay present while starting optical system by human operator)
    
    # Importing CSV files (Timing, ADC and general info)
    timing_adc_dir =  os.path.join(Data_dir, Data_dir_list[Data_dir_list.index('data-timing.csv')])
    # Timing and ADC data
    df_timing_adc = pd.read_csv(timing_adc_dir, skiprows = skip_opticalDelay, dtype = np.single)
    arr_timing_adc = df_timing_adc.to_numpy()
    arr_Time = arr_timing_adc[:,0]          # Time in seconds
    arr_ADC = arr_timing_adc[:,1]           # ADC input (CMOS trigger)
    arr_ADC[arr_ADC >= 1] = 5                # Ceiling the ADC data (ideal signal)
    arr_ADC[arr_ADC < 1] = 0                # Flooring the ADC data (ideal signal)
    
    # If experiment was done in chunks (e.g: a b only) -------------------------------
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
        
    #------------------------------------------------------------------------------
    
    # Finding peaks
    arr_ADC_diff = np.diff(arr_ADC)
    arr_ADC_diff[arr_ADC_diff<0] = 0
    arr_Time_diff = np.delete(arr_Time,[-1])
    
    # Plotting Check
    # plt.figure()
    # plt.title('ADC input')
    # plt.xlabel('Time(s)')
    # plt.ylabel('ADC (V)')
    # plt.plot(arr_Time_diff,arr_ADC_diff)
    # # plt.plot(arr_Time,arr_ADC)
    # plt.ion()
    # plt.show()
    
    # Finding peaks
    # timestamp_frame = ( arr_ADC_diff - np.roll(arr_ADC_diff,1) > 2) & (arr_ADC_diff - np.roll(arr_ADC_diff,-1) > 2) # for analog
    timestamp_frame = ( arr_ADC_diff - np.roll(arr_ADC_diff,1) > 0.5) & (arr_ADC_diff - np.roll(arr_ADC_diff,-1) > 0.5) # for digital
    
    # Plotting Check
    # plt.figure()
    # plt.title("Timing data") 
    # plt.xlabel("Seq #") 
    # plt.ylabel(r'$\Delta$ Time (ms)') 
    # plt.plot(arr_ADC_diff)
    # plt.plot(timestamp_frame.nonzero()[0], arr_ADC_diff[timestamp_frame],'ro')
    # plt.show()
    
    # Here I compute the indices of the timestamps 
    timestamp_frame = timestamp_frame.nonzero()[0]                                        # Timestamp indices of the frames (FOIL Camera)
    # sequences
    temp_vec = np.diff(timestamp_frame)
    x = np.argwhere(temp_vec > Fs*0.03)                                                   # Detect sequences
    x = x.astype(int)
    x = np.reshape(x,(len(x),))
    x+=1
    x = np.insert(x,0,0)                                                                  # So that we dont miss the first seq
    timestamp_seq = timestamp_frame[x]
    # trials
    xx = np.argwhere(temp_vec > Fs*1)                                                      # Detect trials
    xx = xx.astype(int)
    xx = np.reshape(xx,(len(xx),))
    xx+=1
    xx = np.insert(xx,0,0)    
    
    # xx = np.delete(xx,-1)   # extra
    
    timestamp_trials = timestamp_frame[xx]
    
    # Actual timestamps of the sequences and trials
    timestamp_seq_times = arr_Time[timestamp_seq]           # in seconds
    timestamp_trials_times = arr_Time[timestamp_trials]     # in seconds
    
    #----------------------------- Plotting ---------------------------------------
    plt.figure()
    plt.plot(arr_Time,arr_ADC)
    plt.plot(timestamp_seq_times,arr_ADC[timestamp_frame[x]]+1,'ro')
    plt.plot(timestamp_trials_times,arr_ADC[timestamp_frame[xx]]+1,'go')
    plt.show()
    
    # Exporting Timestamps of the trial start times:
    tt_export = timestamp_frame[xx]
    export_timestamps_trials = {'empty':[0],'t_trial_start':tt_export}
    savemat(filename_trials_export,export_timestamps_trials)
    
    # Converting each channel into matrix form. One .csv file per channel
    # Each column is a trial
    # Second-last is the ADC trigger input from CMOS
    # Last column is the time
    extra_time = 0.5                         # extra time in sec before the start of the trial and after the end of the trial
    extra_time = int(extra_time*Fs)              # extra time in samples before the start of the trial and after the end of the trial
    # finding the size of the read_time_trial
    xx = np.delete(xx,0)
    xx -= 1                                                     # -------------------- NOTE NOTE NOTE ******* Changed from the correct value of -1
    xx = np.append(xx,xx[-1]+xx[1]-xx[0])
    timestamp_trials_end = timestamp_frame[xx]
    arr_temp = np.empty((num_trials,),dtype=np.single)
    for iter_trial in range(num_trials):
        start_index = int(timestamp_trials[iter_trial])                        # start index 
        # end_index = int(timestamp_seq[(iter_trial+1)*num_seq-1] + 140)       # end index (120 samples in 6 ms. This includes the last frame as well.)
        end_index = int(timestamp_trials_end[iter_trial] + 5e-3*Fs)            # end_index (100 samples in 5 ms)
        arr_temp[iter_trial] = end_index - start_index
    stat_median_trial_time = np.median(arr_temp)*1/Fs                                    # Median time of one trial
    stat_median_seq_time = np.median(np.diff(timestamp_seq_times))                       # Median time of one seq
    smallest_duration_trial = int(np.median(arr_temp))                                  # The length of every trial is fixed at this value (in samples)  
    # smallest_duration_trial = int(np.amin(arr_temp))                                  # The length of every trial is fixed at this value (in samples)       
    # smallest_duration_trial = stat_median_trial_time
    
    # Computing the size of the arrays to be constructed for every trial
    iter_trial = 1
    start_index = int(timestamp_trials[iter_trial-1] - extra_time)
    end_index = int(start_index + smallest_duration_trial + 2*extra_time)
    real_time_trial = arr_Time[start_index:end_index]
    real_time_trial = real_time_trial - real_time_trial[0]                              # Actual time (common to each trial)
    real_ADC_trial = arr_ADC[start_index:end_index]                                     # Actual ADC value (common to each trial)
    trial_arr = np.empty((end_index-start_index,num_trials+2),dtype = np.single)        # 2 columns extra (see above)
    trial_arr[:,-1] = real_time_trial                                                   # Time has been stored
    trial_arr[:,-2] = real_ADC_trial                                                    # ADC voltages have been stored
    
    # ---------------------- Amplifier Data Binning ----------------------------
    
    # Names for columns for the exported .csv file
    list_titles = []
    for i in range(num_trials):
        list_titles.append('Trial' + str(i+1))
    list_titles.append('ADC Trigger')
    list_titles.append('Time')
    # Reading each channels file and binning into trials 
    # This will create "Num_chan" .csv files
    iter_chan = 1
    del Data_dir_list[-1]
    for file in Data_dir_list:
        filename_csv = Data_dir_list[iter_chan-1]             # Name of file will be exactly the same as the name of the original file
        filename_csv = os.path.join(output_dir,filename_csv)
        df_ChanData = pd.read_csv(os.path.join(Data_dir,file),skiprows = skip_opticalDelay, dtype = np.single)
        arr_ChanData = df_ChanData.to_numpy()
        arr_ChanData = arr_ChanData[:,0]
        
        for iter_trial in range(num_trials):
            start_index = int(timestamp_trials[iter_trial] - extra_time)                  # start index (start of trial - 0.5 sec)
            # end_index = int(timestamp_seq[(iter_trial+1)*num_seq-1] + extra_time)       
            end_index = int(start_index + smallest_duration_trial + 2*extra_time)         # end index   (end of trial + 0.5 sec)
            trial_arr[:,iter_trial] = arr_ChanData[start_index:end_index]
        df_Chan = pd.DataFrame(trial_arr, columns = list_titles)
        df_Chan.to_csv(filename_csv, mode = 'w', header = True, index = False)
        print('Channel data has been exported to: ',filename_csv)
        print(iter_chan/len(Data_dir_list) * 100,'% done')
        iter_chan += 1
        
    
    # Export exp_summary.xlsx : 
    data_summary = {'Sequence Time(s)':[stat_median_seq_time],'Stimulation Start Time(s)':[stim_start_t + extra_time/Fs]}
    df = pd.DataFrame(data_summary)
    append_df_to_excel(exp_summary_dir,df, sheet_name='Sheet1', startrow=4, index = False)          # appends into excel files conveniently    
        
    try:
        shutil.rmtree(Data_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
# TO DO:
# Add checks to see if there was any error














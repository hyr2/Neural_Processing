def read_stimtxt(matlabTXT):
    # Read .txt file
    Txt_File = open(matlabTXT,"r")
    str1 = Txt_File.readlines()
    stim_start_time = float(str1[3]) - 0.05       # Stimulation start time (50 ms error in labview)
    stim_num = int(str1[6])                 # Number of stimulations
    seq_period = float(str1[4])             # Period of each sequence consisting of 10 to 15 frames
    len_trials = int(str1[1])               # Length of a single trial in # seq
    num_trials = int(str1[7])               # total number of trials
    FramePerSeq = int(str1[2])              # frames per sequence. Important info to read the data.timing file
    total_seq = num_trials * len_trials
    len_trials_arr = list(range(1,total_seq,len_trials))
    print('Each sequence lasts...',seq_period,'sec')
    print('Number of sequences in each trial...',len_trials)
    print('Total number of trials in this dataset are:', num_trials)
    print('Reading the total number of sequences in this dataset...',total_seq,'\n.\n.\n.')
    Txt_File.close()
    
    return stim_start_time, stim_num, seq_period, len_trials, num_trials, FramePerSeq, total_seq, len_trials_arr

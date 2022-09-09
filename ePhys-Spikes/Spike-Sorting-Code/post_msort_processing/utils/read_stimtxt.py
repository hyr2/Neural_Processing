def read_stimtxt(matlabTXT):
    # Read .txt file
    Txt_File = open(matlabTXT,"r")
    str1 = Txt_File.readlines()
    stim_start_time = float(str1[3])        # Stimulation start time
    stim_num = int(str1[6])                 # Number of stimulations
    seq_period = float(str1[4])             # Period of each sequence consisting of 10 to 15 frames
    # len_trials = int(str1[1]) + 2           # Length of a single trial in # seq
    num_trials = int(str1[7])               # total number of trials
    # FramePerSeq = int(str1[2])              # frames per sequence. Important info to read the data.timing file
    # total_seq = num_trials * len_trials
    # len_trials_arr = list(range(1,total_seq,len_trials))
    n_seq_per_trial = int(str1[1])
    stim_duration = stim_num * seq_period
    trial_duration = n_seq_per_trial * seq_period
    Txt_File.close()
    return trial_duration, num_trials, stim_start_time, stim_duration
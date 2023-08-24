% Image Registration for IOS images - Preprocessing stage 0

% source_dir
% |
% |__Raw/
% |__whisker_stim.txt


source_dir = input('Please enter the source directory:\n','s');
wv_3 = input('The 3rd wavelength was 510nm or 632nm ? \n','s');
wv_3 = int16(str2num(wv_3));
Raw_dir = fullfile(source_dir,'Raw');
files_raw = dir_sorted(fullfile(Raw_dir, '*.tif'));
out_dir_blue = fullfile(Raw_dir,'480nm');
out_dir_amber = fullfile(Raw_dir,'580nm');
mkdir(out_dir_blue);
mkdir(out_dir_amber)
if wv_3 == 510
    out_dir_wv3 = fullfile(Raw_dir,'510nm');
elseif wv_3 == 632
    out_dir_wv3 = fullfile(Raw_dir,'632nm');
else
    error('Select a correct wavelength for wv_3! \n');
end
mkdir(out_dir_wv3);

whiskerStim_txt = fullfile(source_dir,'whisker_stim.txt');

path_parameter_file = '/home/hyr2/Documents/MATLAB/Elastix/Parameters_Affine.txt';

% Experimental Parameters:
number_wavelength = 3;          % Enter Number of wavelengths here
[len_trials,FramesPerSeq, stim_start_time, seq_period, stim_num, num_trials] = read_whiskerStimTxT(whiskerStim_txt);
len_trials_all_Lambda = number_wavelength * len_trials;
num_frames = len_trials * num_trials * number_wavelength;
start_trial = [1:len_trials_all_Lambda:num_trials*len_trials_all_Lambda];

% Binning images by wavelength:
start_trial = [1:len_trials_all_Lambda:num_trials*len_trials_all_Lambda];

for iter_trial = 1:num_trials
temp_start = start_trial(iter_trial); 
temp_start_amber = temp_start + 1;
temp_start_wv3 = temp_start + 2;
    for iter_seq = 1:len_trials
        index = temp_start + (iter_seq-1)*number_wavelength;
        file = files_raw(index).name;
        filePath = fullfile(Raw_dir,file);
        filePath_dst = fullfile(out_dir_blue,file);
        [~,~] = movefile(filePath,filePath_dst);
        
        index = temp_start_amber + (iter_seq-1)*number_wavelength;
        file = files_raw(index).name;
        filePath = fullfile(Raw_dir,file);
        filePath_dst = fullfile(out_dir_amber,file);
        [~,~] = movefile(filePath,filePath_dst);
        
        index = temp_start_wv3 + (iter_seq-1)*number_wavelength;
        file = files_raw(index).name;
        filePath = fullfile(Raw_dir,file);
        filePath_dst = fullfile(out_dir_wv3,file);
        [~,~] = movefile(filePath,filePath_dst);
    end
end

    
IOS_align_elastix(out_dir_blue,1,'parameter_file',path_parameter_file,'out_dir',Raw_dir);
IOS_align_elastix(out_dir_amber,1,'parameter_file',path_parameter_file,'out_dir',Raw_dir);
IOS_align_elastix(out_dir_wv3,1,'parameter_file',path_parameter_file,'out_dir',Raw_dir);
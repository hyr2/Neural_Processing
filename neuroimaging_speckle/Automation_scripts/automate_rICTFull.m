%% Read txt files
source_dir = pwd;

% source_dir = 'H:\Data\10-27-2021\10-27-2021\data-b';
file_sc = fullfile(source_dir, 'data' , 'CompleteTrials_SC');
matlabTxT = fullfile(source_dir, 'extras' , 'whisker_stim.txt');
output_dir = fullfile(source_dir,'Processed');



[len_trials,FramesPerSeq, stim_start_time, seq_period, stim_num, num_trials] = read_whiskerStimTxT(matlabTxT);

delta_t = seq_period; 
trials = num_trials;
output_dir_alpha = fullfile(output_dir,['1-',num2str(trials)]);

AverageSC(file_sc,source_dir);

rICT_Full(output_dir_alpha,source_dir,delta_t,'trials',trials,'video_range',[1.08 1.14],'vessel_flag',false,'ROI','H:\Data\10-27-2021\10-27-2021\data-b\Processed_Full-Analysis_old\data.mat');

% Saving individual frames
file_loc = fullfile(pwd,'Processed_Full-Analysis','ICT.mp4');
file_dst =  fullfile(pwd,'Processed_Full-Analysis','frames');
mkdir(file_dst);
obj = VideoReader(file_loc);
vid = read(obj);
frames = obj.NumberOfFrames;
for x = 1 : frames
    filename = strcat('frame-',num2str(x),'.png');
    filename = fullfile(file_dst,filename);
    imwrite(vid(:,:,:,x),filename);
end
clear 
close all;
% Partial Automation (ie batch process) IOS imaging files
parent_directory = 'C:\Data\RH-8';
file_X = dir_sorted(parent_directory);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));

global source_dir

for iter_filename = file_X
    % selecting ROIs for all folders first before running the processing
    % loop
    iter_filename = string(iter_filename);
    source_dir = fullfile(parent_directory,iter_filename);
    Raw_dir = fullfile(parent_directory,iter_filename,'Raw');
    [BW,mask,save_flag] = batch_ROI(source_dir,Raw_dir,0);
    if save_flag == 1
        mat_dir = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        save(mat_dir,'BW','mask');
    end
end

for iter_filename = file_X
    iter_filename = string(iter_filename);
    source_dir = fullfile(parent_directory,iter_filename);
    IOS_process_1;
    clearvars -except file_X parent_directory iter_filename;
    close all;
end

% source_dir = 'C:\Data\RH-8\12-5-22';
% IOS_process_1
% clear;
% close all;
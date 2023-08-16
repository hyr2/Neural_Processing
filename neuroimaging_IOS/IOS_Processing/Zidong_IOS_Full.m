clear 
close all;
% Partial Automation (ie batch process) IOS imaging files
parent_directory = '/home/hyr2-office/Documents/Data/IOS_imaging/rh7/';
file_X = dir_sorted(parent_directory);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));

global source_dir

%{
% selecting ROIs for all folders first before running the processing
% loop
for iter_filename = file_X
    iter_filename = string(iter_filename);
    source_dir = fullfile(parent_directory,iter_filename);
    disp(source_dir);
    Raw_dir = fullfile(parent_directory,iter_filename,'Raw');
    [BW,mask,coord_r,save_flag] = batch_ROI(source_dir,Raw_dir,0);
%     [BW,mask,coord_r,save_flag] = batch_ROI(source_dir,Raw_dir,0);
    if save_flag == 1
        mat_dir = fullfile(source_dir,'Processed','mat_files','ROI.mat');
        save(mat_dir,'BW','mask','coord_r');
%         save(mat_dir,'coord_r',"-append");
    end
end

% Running IOS_process_z1
for iter_filename = file_X
    iter_filename = string(iter_filename);
    source_dir = fullfile(parent_directory,iter_filename);
    IOS_process_z1;
    clearvars -except file_X parent_directory iter_filename;
    close all;
end
%}

% Running IOS_process_z2
for iter_filename = file_X
    iter_filename = string(iter_filename);
    source_dir = fullfile(parent_directory,iter_filename);
    IOS_process_z2;
    clearvars -except file_X parent_directory iter_filename;
    close all;
end

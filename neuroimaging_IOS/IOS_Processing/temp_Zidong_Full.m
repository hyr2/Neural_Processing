% temp Zidong
% This code is a temporary script that I used because I missed the saving
% of an averaged image in the processed folder. I also missed reference
% coordinate labelling. This code takes care of these two things and saves
% them in the local drive

clear 
close all;
% Partial Automation (ie batch process) IOS imaging files
parent_directory = '/home/hyr2-office/.media/share/xl_neurovascular_coupling/November_2021 - NVC/RH-9';         % Reading from Remote Server
parent_directory_local = '/home/hyr2-office/Documents/Data/IOS_imaging/rh9/';           % Saving into local PC. Must have the same sessions IDs as the Server
file_X = dir_sorted(parent_directory);
file_X = {file_X.name};
file_X =  file_X(~ismember(file_X,{'.','..'}));
cam_in = 0;


for iter_filename = file_X
    iter_filename = string(iter_filename);
    source_dir = fullfile(parent_directory,iter_filename);
    Raw_dir = fullfile(parent_directory,iter_filename,'IOS','Raw');
    disp(source_dir);
    [sample_img,coord_r,save_flag] = temp_batch_ROI(source_dir,Raw_dir,cam_in);

    if save_flag == 1
        mat_dir = fullfile(parent_directory_local,iter_filename,'Processed','mat_files','ROI.mat');
        Avg_directory = fullfile(parent_directory_local,iter_filename,'Processed','Average','sample_image.mat');
        save(Avg_directory,'sample_img');
        save(mat_dir,'coord_r',"-append");
    end

end



function [BW,mask,save_flag] = batch_ROI(source_dir,Raw_dir,cam_in)
%BATCH_ROI Function is used to ask the user for drawing the ROIs
%   INPUTS:
%       source_dir : session ID as folder name 
%       Raw_dir : directory containing the .tiff files for the session
%       cam_in : camera flag. Is this the Andor Camera? [1,0] 0: Hamamatsu Camera (installed on 2022-08)

save_flag = 0;
mat_dir = fullfile(source_dir,'Processed','mat_files');
if ~isfolder(mat_dir)
    mkdir(mat_dir);
else    % exit function if ROI.mat already exists
    if isfile(fullfile(mat_dir,'ROI.mat'))
        BW = 0;
        mask = 0;
        return
    end
end
% %% ROI Selection
files_raw = dir_sorted(fullfile(Raw_dir, '*.tif'));
iter_local_local = 1;       
vec_img = [2:3:29];      % averaging 580nm image for a good underlay image
for iter_local = vec_img   
    thisImage = double(imread(fullfile(Raw_dir,files_raw(iter_local).name)));
    if iter_local_local
        sumImage = thisImage;
    else
        sumImage = sumImage + thisImage;
    end
    iter_local_local = iter_local_local + 1;
end
sumImage = sumImage / length(vec_img);
sample_img = sumImage;
sample_img = mat2gray(sample_img);
if cam_in == 1
    sample_img = image_transform_B(sample_img);
else
    sample_img = image_transform_A(sample_img);
end
amax = max(sample_img(:));
amin = min(sample_img(:));
mask = drawROIs(sample_img,[amin,amax]);
% Define a craniotomy
BW = defineCraniotomy(sample_img,[amin amax]);
% mat_dir = fullfile(mat_dir,'ROI.mat');
% save(fullfile(mat_dir,'ROI.mat'),'BW','mask');
save_flag = 1;
end


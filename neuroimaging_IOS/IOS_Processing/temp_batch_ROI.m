function [sample_img,coord_r,save_flag] = temp_batch_ROI(source_dir,Raw_dir,cam_in)

% Temporary script goes with temp_Zidong_Full (read description)

save_flag = 0;
%% ROI Selection
files_raw = dir_sorted(fullfile(Raw_dir, '*.tif'));
iter_local_local = 1;       
vec_img = [2:3:49];      % averaging 580nm image for a good underlay image
for iter_local = vec_img   
    thisImage = double(imread(fullfile(Raw_dir,files_raw(iter_local).name)));
    if iter_local_local == 1
        sumImage = thisImage;
    else
        sumImage = sumImage + thisImage;
    end
    iter_local_local = iter_local_local + 1;
end
sumImage = sumImage / length(vec_img);  % Average of multiple images
sample_img = sumImage;
sample_img = mat2gray(sample_img);
if cam_in == 1
    sample_img = image_transform_B(sample_img);
else
    sample_img = image_transform_A(sample_img);
end
amax = max(sample_img(:));
amin = min(sample_img(:));
std_local = std(sample_img(:));
mean_local = nanmean(sample_img(:));

% Define coordinate for centroid tracking
fig = figure('Name', 'Select location for Centroid', 'NumberTitle', 'off','WindowState','maximized');
imshow(sample_img,[mean_local-2*std_local,mean_local+2*std_local]);
[x_r,y_r] = getpts(fig);
coord_r = [x_r,y_r];  % reference coordinates
close all;
% set(fig,'WindowState','maximized');
save_flag = 1;
end


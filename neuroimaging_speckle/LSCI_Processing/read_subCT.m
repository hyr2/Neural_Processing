function [avg_CT, count] = read_subCT(files)

% Reads all CT .mat files from a given folder and averages them.
% This function can be useful to compute the average baseline 
% from a list of baseline CT time bins carefully placed in a
% folder.
%
% Inputs: 1) location of directory containing the .mat files
% Outputs: 1) 2D array of average CT
%          2) Number of CT frames used to find the average CT frame

folder = files(1).folder;
str_file = fullfile(folder, files(1).name);
load(str_file,'avg');
dim_CT = size(avg);
total_img = length(files);
avg_CT = zeros(dim_CT(1),dim_CT(2),'double');
for i = 1:total_img
    str_file = fullfile(folder, files(i).name);
    load(str_file,'avg');
    avg_CT = avg_CT + avg;
end
avg_CT = avg_CT/total_img;
count = total_img;
end

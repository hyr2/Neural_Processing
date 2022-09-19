clear all;
close all;
% File location for the SC files
file_sc = 'E:\Whisker Stimulation\2-22-2021\SC_images\first 50';
% SC range
sc_range = [0.02 0.325];

files = dir_sorted(fullfile(file_sc, '*.sc'));
s = size(files);
trials = s(1)/49;
sc = read_subimage(files, -1 , -1, 1:s(1));
sc = mean(sc, 3)';
F = SpeckleFigure(sc, sc_range);

% Setting the resolution of the image
resolution = get(0, 'screenpixelsperinch');

filename_png = fullfile(pwd,'results','averaged');
filename_sc_std = fullfile(pwd,'results','stdE.sc');
filename_sc = fullfile(pwd,'results','averaged.sc');
write_sc(sc',filename_sc);
F.savePNG(filename_png, resolution);

% Computing the standard error of the mean
std = zeros(size(sc));
for i = [1:trials]
    sc_i = read_subimage(files, -1 , -1, i);
    sc_i = mean(sc_i,3)';
    temp = (sc_i - sc).^2;
    std = temp + std;
end
std = std/(trials-1);
std = sqrt(std);
std = std/sqrt(trials);

write_sc(std,filename_sc_std);

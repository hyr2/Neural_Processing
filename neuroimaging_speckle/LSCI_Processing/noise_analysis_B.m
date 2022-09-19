clear all;
close all;
% File location for the SC files
file_sc = 'E:\Whisker Stimulation\2-22-2021\Binned\first-50\stdE';
files = dir_sorted(fullfile(file_sc, '*.sc'));

stdE = read_subimage(files, -1 , -1, 2);
stdE = mean(stdE, 3)';

file_sc = 'E:\Whisker Stimulation\2-22-2021\Binned\first-50\SC_images';
files = dir_sorted(fullfile(file_sc, '*.sc'));
s = size(files);

mkdir('SC_images_minus');
mkdir('SC_images_plus');
for i = [1:s(1)]
    name_sc = ['Bin', num2str(i,2), '.sc'];    % name of file
    sc = read_subimage(files, -1 , -1, i);
    sc = mean(sc, 3)';
    sc_plus = sc + stdE;
    sc_minus = sc - stdE;
    filename_sc = fullfile(pwd,'SC_images_minus',name_sc);
    write_sc(sc_minus',filename_sc);
    filename_sc = fullfile(pwd,'SC_images_plus',name_sc);
    write_sc(sc_plus',filename_sc);
end

function Mat2SC(file_sc,file_dr)
%MAT_2_SC Summary of this function goes here
%   Detailed explanation goes here
files = dir_sorted(fullfile(file_sc, '*.mat'));
folder = files(1).folder;
num = length(files);

% convert to .SC
im_ext = '.sc';
mkdir(file_dr);
for iter = 1:num
    str_file = fullfile(folder, files(iter).name);
    name_sc = ['data','.', num2str(iter,'%05d'), im_ext];
    filename_sc = fullfile(file_dr,name_sc);
    load(str_file,'SC_REG');
    write_sc(SC_REG',filename_sc);
end

end


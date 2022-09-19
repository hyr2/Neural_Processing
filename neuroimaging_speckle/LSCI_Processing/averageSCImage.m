file_sc = '/media/luanlab/Data_Processing/Speckle4Haad/07-26-aged/9-26-2021';

% used to create an averaged .sc image 

files = dir_sorted(fullfile(file_sc, '*.sc'));
SC = read_subimage(files,-1,-1,1:100);
SC = mean(SC,3,'omitnan')';
write_sc(SC,'averaged_SC.sc');

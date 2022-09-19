file_sc = 'H:\Data\temp';

% used to create an averaged .sc image 

files = dir_sorted(fullfile(file_sc, '*.sc'));
SC = read_subimage(files,-1,-1,1:100);
SC = mean(SC,3,'omitnan')';
write_sc(SC,'averaged_SC.sc');

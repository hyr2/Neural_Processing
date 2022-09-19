file_sc = 'H:\Data\6-5-2021\a\data\CompleteTrials_SC';

files = dir_sorted(fullfile(file_sc,'*.sc'));

SC = read_subimage(files,-1,-1,[1,260,519,778]);

SC1 = SC(:,:,1);
SC1 = SC1';
imshow(SC1,[0.02 0.35]);

function AverageSC(file_sc,outputdir)

% used to create an averaged .sc image 
files = dir_sorted(fullfile(file_sc, '*.sc'));
SC = read_subimage(files,-1,-1,1:100);
SC = mean(SC,3,'omitnan')';


output_dir = fullfile(outputdir,'averaged_SC.sc');
output_dir1 = fullfile(outputdir,'averaged_SC.tiff');
write_sc(SC,output_dir);

% normalize SC image
delta = 0.6 - 0.01;
SC1 = SC/delta;
imwrite(SC1,output_dir1,'tiff');

end


% Thesis Ch2 
% Raw speckles

folder_in  = '/home/hyr2-office/Documents/Data/NVC/Main-Animals/processed_data_bbc8/LSCI';
files = dir_sorted(fullfile(folder_in, 'data.0*'));

I = read_subimage(files,186:835 ,88:801,1:5:40);
num_frames = size(I,3);



for iter = 1:num_frames
    fg = figure;
    subplot(1,2,1);

    image_iter = I(:,:,iter);
    imshow(image_iter,[0,255]);
    subplot(1,2,2);

    sc = speckle_contrast(image_iter,7);
    imshow(sc,[0.05,0.45]);
    
    fg.Position = [0,0,500,500];

    saveas(gcf,fullfile(folder_in,strcat(num2str(iter),'.png')))
    close all;
end



I = read_subimage(files,186:835 ,88:801,1:40);
num_frames = size(I,3);
xx = 5;

[sc,~,~] = temporal_speckle_contrast2(I,xx);

for iter = 1:num_frames/xx
    fg = figure;

    imshow(sc(:,:,iter),[0.05,0.45]);
    
    fg.Position = [0,0,500,500];

    saveas(gcf,fullfile(folder_in,strcat('temporal_',num2str(iter),'.png')))
    close all;
end
close all;

file_sc = 'G:\Whisker Stimulation\FH-BC2\6-18-2021\CBF\awake\data';
file_sc = dir_sorted(fullfile(file_sc,'*.sc')); % baseline
SC = read_subimage(file_sc);

load('H:\LSCI\Speckle_Processing\Processed_Full-Analysis\rICT.mat');
load('G:\Whisker Stimulation\FH-BC2\6-18-2021\CBF\awake\Processed_Full-Analysis-main\data.mat','ROI')
ROI = ROI(:,:,2);

F = SpeckleFigure(SC, [0.02 0.42], 'visible', false);
% F.showOverlay(zeros(size(SC)), [0 0.04], zeros(size(SC)),'use_divergent_cmap', true);
% F.showScalebar();
avg(avg < 0.97) = 0.97;
avg(avg > 1.16) = 0.97;
avg(avg > 1.0 & avg < 1.0425) = 0.97;
avg(avg == NaN) = 0.97;
% avg(:,[500:end]) = 0.97;



alpha = 0.3*ones(size(SC));  
F.showOverlay(avg,[0.98 1.09], alpha,'use_divergent_cmap', true);



% vid
output_dir = 'H:\LSCI\Speckle_Processing\Processed_Full-Analysis\rICT.mp4'
fps = 1;
v = VideoWriter(output_dir, 'MPEG-4');
v.FrameRate = fps;
open(v);

writeVideo(v, F.getFrame());
writeVideo(v, F.getFrame());
writeVideo(v, F.getFrame());

close(v);
ReadFrames_Video
% figure;
% ha = imshow(SC,[0.02 0.45],'Colormap',gray(256));
% % showMaskAsOverlay(0.4,ROI,'g',[],0);
% 
% 
% cmap = colormap(diverging_map(linspace(0,1,256), [0.230, 0.299, 0.754], [0.706, 0.016, 0.150]));
% avg = avg - 0.98;
% avg(avg<0) = 0;
% avg(avg>0.085) = 0;
% avg = mat2gray(avg,[0 1]);
% avg(avg<0.069) = 0;
% avg = imgaussfilt(avg,0.5);
% % showMaskAsOverlay(0.2,avg,'r',[],0);
% 
% hb = imshow(avg,[0 0.08],'Colormap',gray(128))
% hb.AlphaData = 0.5;
% saveas(gcf,'Speckle_overlay.eps','eps');
% saveas(gcf,'Speckle_overlay.svg','svg');
% saveas(gcf,'Speckle_overlay.png','png');
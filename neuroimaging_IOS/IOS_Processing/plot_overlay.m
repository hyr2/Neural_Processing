function plot_overlay(sample_img,delRR,overlayrange,contrast_levels,selpathOut,name_pic)

amin = contrast_levels(1);
amax = contrast_levels(2);
c_limits = overlayrange;

F = SpeckleFigure(sample_img, [amin,amax], 'visible', false);
F.showOverlay(zeros(size(delRR)),c_limits, zeros(size(delRR)),'use_divergent_cmap', false);
% Generate the overlay alpha mask excluding values outside overlay range
alpha = 0.5*ones(size(delRR)); % transparency
alpha(delRR < c_limits(1)) = false;
alpha(delRR > c_limits(2)) = false;
%alpha = alpha & BW;
% Update the figure
F.updateOverlay(delRR, alpha);
name_pic = strcat(name_pic ,'_neg.bmp');
outputFileName = fullfile(selpathOut,name_pic);
F.saveBMP(outputFileName,200);


end
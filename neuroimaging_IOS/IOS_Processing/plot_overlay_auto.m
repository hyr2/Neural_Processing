function plot_overlay_auto(sample_img,delRR,overlayrange,selpathOut,name_pic)
% Plots overlay of activation region over a raw image of the brain

sample_img = adapthisteq(sample_img,'Distribution','rayleigh','Alpha',0.5);
c_limits = overlayrange;

F = SpeckleFigure(sample_img, [0,1] , 'visible', false);
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
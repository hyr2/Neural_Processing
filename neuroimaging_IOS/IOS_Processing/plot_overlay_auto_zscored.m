function plot_overlay_auto_zscored(sample_img,delRR_z,selpathOut,name_pic)
% Plots overlay of activation region over a raw image of the brain


sample_img = adapthisteq(sample_img,'Distribution','rayleigh','Alpha',0.625);
F = SpeckleFigure(sample_img, [0,1] , 'visible', true);
F.showOverlay(zeros(size(delRR_z)),[0,1], zeros(size(delRR_z)),'use_divergent_cmap', false);

% Generate the overlay alpha mask excluding values outside overlay range
alpha = 0.5*ones(size(delRR_z)); % transparency
alpha(delRR_z <= 0) = false;
alpha(delRR_z > 1) = false;
%alpha = alpha & BW;
% Update the figure
F.updateOverlay(delRR_z, alpha);
F.hideColorbar();
name_pic = strcat(name_pic ,'_neg.bmp');
outputFileName = fullfile(selpathOut,name_pic);
F.saveBMP(outputFileName,200);
F.delete();


end
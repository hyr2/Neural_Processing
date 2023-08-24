F = SpeckleFigure(avg_frame_local, [10000,50000], 'visible', true);
F.showOverlay(zeros(size(delRR)), [-0.025 0], zeros(size(delRR)),'use_divergent_cmap', false);
% Generate the overlay alpha mask excluding values outside overlay range
alpha = 0.5*ones(size(delRR)); % transparency
alpha(delRR < -0.025) = false;
alpha(delRR > 0.0) = false;
%alpha = alpha & BW;
% Update the figure
F.updateOverlay(delRR, alpha);
F.saveBMP(,200);
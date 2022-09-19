function rICT_Figure(SC,r,rICT,alpha,cutoff)
%rICT_Figure Generates an rICT overlay figure
% F = rICT_Figure(SC,r,rICT,alpha,cutoff) Generates a figure overlaying
% rICT on speckle contrast imagery with a user-defined threshold
% controlling the alpha transparency of the rICT layer.
%

F = gcf;

% Overlay rICT on SC
A1 = axes('Parent', F);
A2 = axes('Parent', F);
imshow(SC, r, 'Parent', A1);
I2 = imshow(rICT, 'Parent', A2);
set(I2, 'AlphaData', alpha);

% Define colorbar
caxis(cutoff);
colormap(A2, flipud(pmkmp(256, 'CubicYF')));
c = colorbar;
set(c,'YTick',cutoff(1):0.1:cutoff(2));
ylabel(c, 'Relative ICT');

end

function [im_delta_c] = HyperSpectral_haemoglobin_track_BYG(im_blue,im_yellow,im_green)
% HyperSpectral_haemoglobin_track: This function uses the reflectance change for specific
% wavelengths and computes the change in the concentration of the haemoglobins
% (oxygenated and deoxygenated). The experiment was performed by illuminating
% the surface of the brain with 10 nm FWHM LED lights of different wavelengths 
% in a trial based experiment. We therefore have a time stack of iamges. A 
% baseline time is selected and using that baseline, changes in intensities 
% are calculated (pixel-by-pixel)

%   INPUT PARAMETERS:
%       im_blue: I(t)/I(t_0) = (deltaR/R + 1). A 2D image matrix.
%       im_yelloow: I(t)/I(t_0) = (deltaR/R + 1). A 2D image matrix.
%       im_red: I(t)/I(t_0) = (deltaR/R + 1). A 2D image matrix.
%       X_blue: Differential path length factor at blue for the brain
%       X_yellow: Differential path length factor at blue for the brain
%       X_red: Differential path length factor at blue for the brain
%   OUTPUT PARAMETERS:
%       im_delta_c: The change in concentrations of HbO and HbT relative to
%       baseline. Computation based on the input deltaR/R.

% Definitions:
% e: molar extinction coefficient (function of wavelength of course) [units: cm^-1 M^-1]
% X: Differntial path length factor (almost same as path length but not
% exactly) [units: cm]


% Lan-Luan Experimental Lab parameters:
% blue: 480 nm / 10 nm (e_HbO: 26629.2, e_HbR: 14550, X: 0.064898)
% green: 532 nm / 10 nm (e_HbO: 43876, e_HbR: 40584, X: 0.0338528)   [X]
% yellow: 580 nm / 10 nm (e_HbO: 50104, e_HbR: 37020, X: 0.0306261)  
% red: 632 nm / 10 nm (e_HbO: 561.2, e_HbR: 4930.8, X: 0.3924145)    [X]
% green: 510 nm / 10 nm (e_HbO: 20035.2, e_HbR : 25773.6, X: 0.0712537)     

img_size = size(im_green);
im_delta_c = zeros(img_size(1),img_size(2),2);        % initializing the matrix of concentrations

% Values of molar extinction coefficient 
% obtained from literature 
% (https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frstb.2015.0360&file=rstb20150360supp1.pdf)
e_HbO_blue = 26629.2; e_HbR_blue = 14550;      
e_HbO_yellow = 50104  ; e_HbR_yellow = 37020; 
e_HbO_green = 20035.2; e_HbR_green = 25773.6; 

% Differential path length factors: (using Andrew Dunn's)
X_blue = 0.0838;     
X_yellow = 0.0490;  
X_green = 0.0823;

% Change in mu (absorption coefficient)
delta_mu_a_blue = -1/X_blue * log(im_blue);
delta_mu_a_yellow = -1/X_yellow * log(im_yellow);
delta_mu_a_green = -1/X_green * log(im_green);

A = [e_HbO_blue, e_HbR_blue;
     e_HbO_yellow, e_HbR_yellow;
     e_HbO_green, e_HbR_green];

for x = [1:img_size(1)]
    for y = [1:img_size(2)]
        Y = [delta_mu_a_blue(x,y);
             delta_mu_a_yellow(x,y);
             delta_mu_a_green(x,y)];
         b = leastSquare_direct(A,Y);
         im_delta_c(x,y,:) = b;
    end
end
end


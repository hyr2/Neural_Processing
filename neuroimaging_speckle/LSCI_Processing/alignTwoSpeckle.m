function [aligned,mask] = alignTwoSpeckle(moving,fixed,varargin)
%alignTwoSpeckle Performs image registration on two SC images
%   aligned = alignTwoSpeckle(moving,fixed) performs image registration on two
%   speckle contrast images such that moving is aligned to fixed. A hybridized
%   image is returned combining the registered moving and fixed images.
%
%   [aligned,mask] = alignTwoSpeckle(moving,fixed) performs the image
%   registration described above and returns a binary mask describing how
%   the moving image was manipulated to fit the fixed image.
%
%   alignTwoSpeckle(moving,fixed,p) performs image registration using the
%   user-defined Elastix parameter file located at path p.
%
    
    if(nargin >= 3)
        p = varargin{1}; % User-defined parameter file
    else
        p = which('Speckle/elastix_parameters.txt');
    end

    % Equalize the images for the registration
    fixed_scaled = histeq(fixed);
    moving_scaled = histeq(moving);

    % Use Elastix + Transformix register the images
    d = elastix(fixed_scaled,moving_scaled,p);
    [aligned, mask] = transformix(d, moving);
    
    % Create hybrid image using template image to eliminate alignment gap
    aligned = imadd(aligned.*mask, fixed.*~mask);
    
end

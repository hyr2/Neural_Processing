function [image_output] = image_transform_B(image_input)
%IMAGE_TRANSFORM_B Rotates and flips the image horizontally 
%   After applying the transformations, this image will be in the same
%   orientation as the speckle image as well as the 2p image.

% For Andor Camera
image_output=rot90(image_input,-1);
image_output=fliplr(image_output);
image_output=double(image_output);

end


function [image_output] = image_transform_A(image_input)
%IMAGE_TRANSFORM_A Rotates and flips the image horizontally 
%   After applying the transformations, this image will be in the same
%   orientation as the speckle image as well as the 2p image.

% For Hamamatsu Camera
image_output=rot90(image_input);
image_output=fliplr(image_output);
image_output=double(image_output);

end


function [avg, S, count] = stats_images(IM_3D)
%   Get the statistics of a stack of images. Images are expected 
%   to be stacked along dimension 3. The function will output two matrices.
%   1. Mean 
%   2. Standard deviation 
%   of all the images in the stack.
    dimensions = size(IM_3D);
    if length(dimensions) == 3
        count = dimensions(3);
    else
        count = 1;
    end
    S = zeros(dimensions(1),dimensions(2),'double');
    avg = zeros(dimensions(1),dimensions(2),'double');
    avg = mean(IM_3D,3,'omitnan');
    S = std(IM_3D,0,3,'omitnan');
end
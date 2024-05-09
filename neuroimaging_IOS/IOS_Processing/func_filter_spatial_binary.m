function [output_image] = func_filter_spatial_binary(input_image,kernel_size)

h = fspecial('average', [kernel_size kernel_size]);  % This creates a 2x2 averaging filter

output_image = imfilter(input_image, h, 'replicate');

end


function mask = defineCraniotomy(SC,r)
%defineCraniotomy Prompts user to define craniotomy area using speckle contrast image
% mask = defineCraniotomy(SC,r) returns a binary mask corresponding to the
% user-defined outline of the craniotomy area using a speckle contrast
% image for guidance. These masks can be used to exclude data outside of
% the cranial window.
%

% Suppress warnings about figure size
warning('off', 'Images:initSize:adjustingMag');

figure('Name', 'Define the craniotomy area (Press ESC to skip)', 'NumberTitle', 'off');
mask = roipoly(mat2gray(SC, r));
close(gcf);

if(isequal(size(mask), size(SC)))
  mask = logical(mask);
else
  mask = true(size(SC));
end

end

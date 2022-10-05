function [aligned, mask] = transformix(d,moving)
%transformix Wrapper for calling transformix to apply an image transform
% aligned = transformix(d,moving) applies the transform parameter file(s)
% in the directory to the input image and returns the transformed image.
%
% [aligned, mask] = transformix(d,moving) also returns a binary mask
% showing the position of the transformed image relative to the original.
%
% Input Arguments:
%   d = Path to directory output by elastix containing transform file(s)
%   moving = Moving image (Native format, no scaling necessary)
%
% Output Arguments:
%   aligned = Transformed image
%
% Optional Output Arguments:
%   mask = Binary mask of the transformed image compared to the original
%

    % Get the path to the transform file(s)
    % Only the last file will be used by transformix but we need to return
    % all of the files for saving
    files = dir(fullfile(d,'TransformParameters.*.txt'));
    tform = cell(length(files), 1);
    for i = 1:length(files)
        tform{i} = fullfile(files(i).folder, files(i).name);
    end

    % Write unmodified image to file for transformix access
    r = [min(moving(:)) max(moving(:))];
    m = fullfile(d,'moving.tiff');
    imwrite(uint16(mat2gray(moving,r)*65534) + 1,m,'Compression','None');

    % Transform full image using the calculated transform
    CMD = sprintf('transformix -in %s -out %s -tp %s',m,d,tform{end});
    [~,~] = system(CMD);
    aligned = imread(fullfile(d,'result.tiff'));

    % Create mask showing positioning of aligned image
    mask = aligned ~= 0;

    % Convert uint16 back to original scale
    aligned = (double(aligned - 1)/65534)*(r(2) - r(1)) + r(1);    
    
    % Remove the temporary directory
    rmdir(d,'s');
    
end

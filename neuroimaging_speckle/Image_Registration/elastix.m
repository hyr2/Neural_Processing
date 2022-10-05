function d = elastix(fixed,moving,params)
%elastix Wrapper for calling elastix to align two images
% d = elastix(fixed,moving,params) calls the elastix program using the 
% provided parameter file(s) to align moving with fixed. Returns a 
% temporary directory containing the transform files necessary for use with
% transformix.
%
% Input Arguments:
%   fixed = Fixed image (scaled between 0-1)
%   moving = Moving image (scaled between 0-1)
%   params = Path to elastix parameter file(s) (string or cell array)
%       params = '/full/path/to/file.txt'
%       params = {'/full/path/to/file1.txt', '/full/path/to/file2.txt'}
%
% Output Arguments:
%   d = Path to temporary directory containing the output from elastix. The
%   files necessary for applying the transform with transformix are located
%   in this directory.
%

    % Format elastix parameter file(s)
    p = '';
    if iscell(params)
        for i = 1:length(params)
            p = strcat(p, sprintf(' -p "%s"', params{i}));
        end        
    else
       p = sprintf('-p "%s"', params);
    end

    if ispc
        % Create temporary working directory (WINDOWS)
        d = tempname('C:\myTempFiles');
        mkdir(d);
    else
        % Create temporary working directory (Linux/Max)
        d = tempname;
        mkdir(d);
    end


    % Convert fixed image to uint16 and write to file
    fixed = uint16(fixed*65535);
    f = fullfile(d,'fixed.tiff');
    imwrite(fixed,f,'Compression','None');

    % Convert moving image to uint16 and write to file
    moving = uint16(moving*65535);
    m = fullfile(d,'moving.tiff');
    imwrite(moving,m,'Compression','None');
    
    % Register to first frame
    CMD = sprintf('elastix -f %s -m %s -out %s %s',f,m,d,p)
    [~,~] = system(CMD);

end

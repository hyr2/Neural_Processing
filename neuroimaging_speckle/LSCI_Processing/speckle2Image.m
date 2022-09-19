function speckle2Image(file_sc,n_avg,sc_range,varargin)
%speckle2Image Generates image from speckle contrast data
% speckle2Image(file_sc,n_avg,sc_range) load, average, and render speckle 
% contrast data to an image file.
%
%   file_sc = Path to directory of speckle contrast files
%   n_avg = Number of speckle contrast frames to load and average
%   sc_range = Display range for speckle contrast data [min max]
%
% speckle2Image(___,Name,Value) modifies the image appearance using one or 
% more name-value pair arguments.
%
%   'show_scalebar': Toggle display of the scalebar (Default = true)
%       Scale is hard-coded in SpeckleFigure to that of the multimodal system
%
%   'formattype': Specify output file format (Default = 'png')
%       Select either lossy 'png' or lossless 'bmp'
%
%   'resolution': Specify output file resolution (Default = System Resolution)
%       Output image resolution in pixels-per-inch
%

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'n_avg', @isscalar);
addRequired(p, 'sc_range', @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'}));
addParameter(p, 'show_scalebar', true, @islogical);
addParameter(p, 'formattype', 'png', @(x) any(validatestring(x, {'png', 'bmp'})));
addParameter(p, 'resolution', get(0, 'screenpixelsperinch'), @isscalar);
parse(p, file_sc, n_avg, sc_range, varargin{:});

file_sc = p.Results.file_sc;
n_avg = p.Results.n_avg;
sc_range = p.Results.sc_range;
show_scalebar = p.Results.show_scalebar;
formattype = p.Results.formattype;
resolution = p.Results.resolution;

% Read and average the first n speckle contrast frames
files = dir_sorted(fullfile(file_sc, '*.sc'));

sc = read_subimage(files, -1, -1, 1:n_avg);
sc = mean(sc, 3)';

% Create the figure
F = SpeckleFigure(sc, sc_range);

if show_scalebar
  F.showScalebar();
end

% Write the image to file
d_parts = strsplit(files(1).folder, filesep);
filename = fullfile(pwd,d_parts{end});
if strcmp(formattype, 'png')
  F.savePNG(filename, resolution);
elseif strcmp(formattype, 'bmp')
  F.saveBMP(filename, resolution);
end

end
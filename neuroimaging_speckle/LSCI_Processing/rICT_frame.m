function rICT_frame(file_sc,sc_range,varargin)
%rICT_frame Generates single rICT frame
% rICT_frame(file_sc,sc_range) generates an image overlaying relative 
% inverse correlation time (rICT) on speckle contrast data. The rICT is
% calculated with a user-defined timespan as the baseline.
%
%   file_sc = Path to directory of speckle contrast files
%   sc_range = Display range for speckle contrast data [min max]
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS
%
% rICT_frame(file_sc,sc_range,file_sc_baseline) loads separate directory of
% speckle contrast data for the rICT baseline.
%
%   file_sc_baseline = Path to directory of speckle contrast files
%
% rICT_frame(___,Name,Value) modifies the overlay appearance using one or 
% more name-value pair arguments.
%
%   'overlay_range': Set display range for rICT overlay (Default = [0 0.5])
%
%   'use_divergent_cmap': Toggle use of divergent colormap (Default = false)
%       Use divergent red/blue colormap instead of default sequential
%       colormap to visualize rICT.
%
%   'show_scalebar': Toggle display of the scalebar (Default = true)
%       Scale is hard-coded in SpeckleFigure to that of the multimodal system
%
%   'trim_mask': Enable manual trimming of rICT overlay (Default = false)
%
%   'formattype': Specify output file format (Default = 'png')
%       Select either lossy 'png' or lossless 'bmp'
%
%   'resolution': Specify output file resolution (Default = System Resolution)
%       Output image resolution in pixels-per-inch
%
%   'parameter_file': Custom Elastix parameter file (Default = 'Elastix/Parameters_Rigid.txt')
%       Specify full path to custom Elastix parameter file for use during 
%       image registration. Defaults to using the Rigid Transform.
%
%           'Parameters_Translation.txt'
%           'Parameters_Rigid.txt'
%           'Parameters_Affine.txt'
%           'Parameters_BSpline.txt'
%

warning('off', 'Images:initSize:adjustingMag'); % Suppress image size warnings

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
validateFnc = @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'});
addRequired(p, 'sc_range', validateFnc);
addOptional(p, 'file_sc_baseline', '', @(x) exist(x, 'dir'));
addParameter(p, 'overlay_range', [0 0.5], validateFnc);
addParameter(p, 'use_divergent_cmap', false, @islogical);
addParameter(p, 'show_scalebar', true, @islogical);
addParameter(p, 'trim_mask', false, @islogical);
addParameter(p, 'formattype', 'png', @(x) any(validatestring(x, {'png', 'bmp'})));
addParameter(p, 'resolution', get(0, 'screenpixelsperinch'), @isscalar);
addParameter(p, 'parameter_file', which('Elastix/Parameters_Rigid.txt'), @isfile);
parse(p, file_sc, sc_range, varargin{:});

file_sc = p.Results.file_sc;
sc_range = p.Results.sc_range;
file_sc_baseline = p.Results.file_sc_baseline;
overlay_range = p.Results.overlay_range;
use_divergent_cmap = p.Results.use_divergent_cmap;
show_scalebar = p.Results.show_scalebar;
trim_mask = p.Results.trim_mask;
formattype = p.Results.formattype;
resolution = p.Results.resolution;
parameter_file = p.Results.parameter_file;

% Get list of speckle contrast files
files = dir_sorted(fullfile(file_sc, '*.sc'));
file_sc = files(1).folder;
N_total = totalFrameCount(files);

% If baseline data is provided, load all the files in the directory
% Otherwise, prompt user to define baseline
if ~isempty(file_sc_baseline)
  fprintf('Identified %d frames in %s\n', N_total, file_sc);
  [SC1, n] = loadReferenceSC(file_sc_baseline);
else
  t = loadSpeckleTiming(fullfile(file_sc, '*.timing'));
  t = mean(reshape(t,[],N_total), 1);
  t = t - t(1);
  fprintf('Identified %d frames in %s collected over %.2f seconds\n', N_total, file_sc, range(t));
  idx = getBaselineIndex(t);
  n = length(idx);
  SC1 = read_subimage(files, -1, -1, idx);
  SC1 = mean(SC1, 3)';
end

% Load and average the last n frames
SC2 = read_subimage(files, -1, -1, N_total - n + 1:N_total);
SC2 = mean(SC2, 3)';

% If enabled, have user define craniotomy area
if trim_mask
  BW = defineCraniotomy(SC1, sc_range);
else
  BW = true(size(SC1));
end

% Visualize alignment before registration
figure('Name', 'Pre-Registration Frame Alignment', 'NumberTitle', 'off');
imshowpair(mat2gray(SC1, sc_range), mat2gray(SC2, sc_range), 'Scaling', 'joint');

% Register SC2 with SC1
disp('Performing intensity-based image alignment of the two frames');
[SC2_REG, mask] = alignTwoSpeckle(SC2, SC1, parameter_file);
BW(SC1 > sc_range(2)) = false;
BW(SC2_REG > sc_range(2)) = false;

% Visualize alignment after registration
figure('Name', 'Post-Registration Frame Alignment', 'NumberTitle', 'off');
imshowpair(mat2gray(SC1, sc_range), mat2gray(SC2_REG, sc_range), 'Scaling', 'joint');

% Convert SC images to CT and calculate relative ICT image
CT1 = get_tc_band(SC1, 5e-3, 1);
CT2 = get_tc_band(SC2_REG, 5e-3, 1);
rICT = CT1./CT2;

% Generate the overlay alpha mask excluding values outside overlay range
alpha = true(size(rICT));
alpha(rICT < overlay_range(1)) = false;
alpha(rICT > overlay_range(2)) = false;
alpha = alpha & BW & mask;

% Generate the rICT figure
F = SpeckleFigure(SC1, sc_range);
F.showOverlay(rICT, overlay_range, alpha, 'use_divergent_cmap', use_divergent_cmap);

if show_scalebar
  F.showScalebar();
end

% Draw statistics about stroke size
y_lim = get(gca, 'YLim');
x_lim = get(gca, 'XLim');
RES = get(0, 'screenpixelsperinch');
FS = 9 * 72 / RES;
infarct_20 = sum(sum(rICT < 0.2 & BW & mask)) / (335^2); % rICT < 20%
text(x_lim(2) + 20, y_lim(2) - 75,sprintf('<20%%: %.3f mm^2',infarct_20),...
  'FontUnits','Pixels','FontSize',FS,'Color',[0 0 0]);
infarct_50 = sum(sum(rICT < 0.5 & BW & mask)) / (335^2); % rICT < 50%
text(x_lim(2) + 20, y_lim(2) - 50,sprintf('<50%%: %.3f mm^2',infarct_50),...
  'FontUnits','Pixels','FontSize',FS,'Color',[0 0 0]);
infarct_20_70 = sum(sum((rICT > 0.2 & rICT < 0.7) & BW & mask)) / (335^2); % 20% < rICT < 70%
text(x_lim(2) + 20, y_lim(2) - 25,sprintf('20-70%%: %.3f mm^2',infarct_20_70),...
  'FontUnits','Pixels','FontSize',FS,'Color',[0 0 0]);

% Write the image to file
d_parts = strsplit(file_sc, filesep);
filename = fullfile(pwd, [d_parts{end} '_rICT']);
if strcmp(formattype, 'png')
  F.savePNG(filename, resolution);
elseif strcmp(formattype, 'bmp')
  F.saveBMP(filename, resolution);
end

end
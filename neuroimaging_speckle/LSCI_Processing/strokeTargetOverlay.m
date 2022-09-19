function strokeTargetOverlay(bg_image,roi_dir,varargin)
%strokeTargetOverlay Overlays stroke target on speckle image
% strokeTargetOverlay(bg_image,roi_dir) produces a composite image overlaying 
% stroke mask(s) onto speckle image.
%
%   bg_image = Path to background speckle contrast image
%   roi_dir = Path to BMP binary mask(s) output by Speckle Software
%    
% speckle2Image(___,Name,Value) modifies the image appearance using one or 
% more name-value pair arguments.
%
%   'alpha': Set transparency of stroke target overlay (Default = 0.5)
%

p = inputParser;
addRequired(p, 'bg_image', @isfile);
addRequired(p, 'roi_dir', @(x) exist(x, 'dir'));
addParameter(p, 'alpha', 0.5, @isscalar);
parse(p, bg_image, roi_dir, varargin{:});

bg_image = p.Results.bg_image;
roi_dir = p.Results.roi_dir;
alpha = p.Results.alpha;

% Load background image and ROIs
im = imread(bg_image);
ROI = loadROIs(roi_dir);

% Create the figure
F = SpeckleFigure(im, [0 255]);
F.showROIs(ROI, 'stroke_mode', true, 'alpha', alpha);

% Overlay total stroke target area
area = sum(ROI(:)) / (335.^2); % mm^2
YLim = get(gca, 'YLim');
XLim = get(gca, 'XLim');
rectangle(gca, 'Position', [XLim(2) - 220, YLim(2) - 50, 220, 50], ...
  'FaceColor', 'k', 'LineStyle', 'none');
text(gca, XLim(2) - 110, YLim(2) - 28, sprintf('ROI = %0.4fmm^2', area), ...
  'FontUnits', 'pixels', 'FontSize', 12, 'FontWeight', 'Bold', ...
  'Color', [1 - eps, 1, 1], 'HorizontalAlignment', 'center');

% Write to file
[~,name,~] = fileparts(bg_image);
filename = fullfile(pwd, sprintf('%s_Target.png', name));
F.savePNG(filename);

end
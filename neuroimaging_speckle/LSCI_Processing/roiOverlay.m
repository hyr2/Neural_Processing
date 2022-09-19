function roiOverlay(bg_image,roi_dir,varargin)
%roiOverlay Overlays ROIs on speckle image
% roiOverlay(bg_image,roi_dir) produces a composite image overlaying 
% ROI mask(s) onto speckle image.
%
%   bg_image = Path to background speckle contrast image
%   roi_dir = Path to BMP binary mask(s) output by Speckle Software
%    
% speckle2Image(___,Name,Value) modifies the image appearance using one or 
% more name-value pair arguments.
%
%   'alpha': Set transparency of stroke target overlay (Default = 0.5)
%
%   'show_labels': Toggle display of ROI labels (Default = true)
%

p = inputParser;
addRequired(p, 'bg_image', @isfile);
addRequired(p, 'roi_dir', @(x) exist(x, 'dir'));
addParameter(p, 'alpha', 0.5, @isscalar);
addParameter(p, 'show_labels', true, @islogical);
parse(p, bg_image, roi_dir, varargin{:});

bg_image = p.Results.bg_image;
roi_dir = p.Results.roi_dir;
alpha = p.Results.alpha;
show_labels = p.Results.show_labels;

% Load background image
im = imread(bg_image);

% Create the figure
F = SpeckleFigure(im, [0 255]);
F.showROIs(roi_dir, 'alpha', alpha, 'show_labels', show_labels)

% Write to file
[~,name,~] = fileparts(bg_image);
filename = fullfile(pwd, sprintf('%s_Overlay.png',name));
F.savePNG(filename);

end
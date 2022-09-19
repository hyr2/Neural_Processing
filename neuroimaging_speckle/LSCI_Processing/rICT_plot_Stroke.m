function rICT_plot_Stroke(file_sc,n_avg,varargin)
%rICT_plot Generates timecourse plots from speckle contrast data
% rICT_plot(file_sc,n_avg) generates speckle contrast (SC), correlation 
% time (CT), relative inverse correlation time (rICT), and infarct size 
% plots with user-defined ROIs from speckle contrast data. The rICT is 
% calculated with a user-defined timespan as the baseline. The plots are 
% saved as images and the data exported into a .mat file.
%
%   file_sc = Path to directory of speckle contrast files
%   n_avg = Number of speckle contrast frames to load and average
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS
%
% rICT_plot(file_sc,n_avg,file_sc_baseline) loads separate directory of 
% speckle contrast data for the rICT baseline.
%
%   file_sc_baseline = Path to directory of speckle contrast files
%
% rICT_plot(file_sc,n_avg,file_sc_baseline,ROI) use previously-defined ROI 
% masks either loaded from a directory containing exported ROI images from 
% the Speckle Software (*_ccd.bmp format) or passed as an array of binary 
% masks. The file_sc_baseline parameter can be empty (e.g. '' or []).
%
%   ROI = Path to directory of ROI masks from Speckle Software or an array 
%     of binary masks. The mask dimensions must match that of the speckle data.
%
% rICT_plot(___,Name,Value) modifies the data processing using one or more
% name-value pair arguments.
%
%   'sc_range': Set display range for speckle contrast data (Default = [0 0.2])
%       Used when generated ROI overlays
%
%   'scale': Set imaging system resolution in pixels/mm (Default = 402.73)
%       Used when calculating the infarct area. Defaults to the multimodal
%       imaging system resolution.
%
%   'core_threshold': Set rICT threshold for ischemic core (Default = 0.35)
%       Used when calculating the infarct area.
%

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'n_avg', @isscalar);
addOptional(p, 'file_sc_baseline', '', @(x) exist(x, 'dir'));
addOptional(p, 'ROI', '');
validateFnc = @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'});
addParameter(p, 'sc_range', [0.025 0.45], validateFnc);
addParameter(p, 'scale', 402.73, @isscalar);
addParameter(p, 'core_threshold', 0.35, @isscalar);
parse(p, file_sc, n_avg, varargin{:});

file_sc = p.Results.file_sc;
n_avg = p.Results.n_avg;
sc_range = p.Results.sc_range;
file_sc_baseline = p.Results.file_sc_baseline;
ROI = p.Results.ROI;
scale = p.Results.scale;
core_threshold = p.Results.core_threshold;

% Get list of speckle contrast files
files = dir_sorted(fullfile(file_sc, '*.sc'));
N_total = totalFrameCount(files);

% Get timing information
t = loadSpeckleTiming(fullfile(file_sc, '*.timing'));
t = mean(reshape(t,[],N_total), 1);
t_start = getSpeckleStartTime(fullfile(file_sc, '*.log'));

fprintf('Identified %d frames in %s collected over %.2f seconds\n', N_total, files(1).folder, range(t));

% Compute Stroke Area
% read binary mask .bmp file
bm_img = imread('''');
Area_Stoke = Stroke_Area(bm_img,scale);

% If no reference is provided, prompt user to define baseline.
% Otherwise, load all the files in the reference directory and register
% to the first frame of the primary data.
if isempty(file_sc_baseline)
  idx = getBaselineIndex(t);
  SC_REF = mean(read_subimage(files, -1, -1, idx), 3)';
else
  SC_FIRST = mean(read_subimage(files, -1, -1, 1:n_avg), 3)';
  SC_REF = loadReferenceSC(file_sc_baseline, SC_FIRST);
end

% Compute correlation time for baseline
CT_REF = get_tc_band(SC_REF,5e-3,1);

% Reshape data for parallelization
N = floor(N_total/n_avg);
idx = reshape(1:N*n_avg,n_avg,N);
t = reshape(t(1:N*n_avg),[],N);
t = mean(t,1);
t_start = t_start + seconds(t(1));
t = t - t(1);

% Load the final frame for overlay
SC_END = read_subimage(files, -1, -1, idx(:,end));
SC_END = mean(SC_END, 3)';

% If ROI argument is not provided, prompt user to define
if ~isempty(ROI)
  if ischar(ROI)
    ROI = loadROIs(ROI);
  else
    ROI = logical(ROI);
  end
  
  % Check dimensions
  if size(ROI, 1) ~= size(SC_REF, 1) || size(ROI, 2) ~= size(SC_REF, 2)
    error('ROI dimensions (%dx%d) do not match speckle data (%dx%d)', size(ROI,1), size(ROI, 2), size(SC_REF, 1), size(SC_REF, 2));
  end
  
else
  ROI = drawROIs(SC_REF, sc_range);
end

% Introduce NaN for computational purposes
ROI_NaN = double(ROI);
ROI_NaN(ROI_NaN == 0) = NaN;

% Generate name labels for the ROIs
N_ROI = size(ROI,3);
names = generateROILabels(N_ROI);

% Calculate baseline CT values within each ROI
CT_REF_MASK = zeros(1,N_ROI);
for i = 1:N_ROI
  CT_MASKED = ROI_NaN(:,:,i).*CT_REF;
  CT_REF_MASK(i) = nanmean(CT_MASKED(:));
end

% Preallocate Storage
SC = zeros(N,N_ROI);
CT = zeros(N,N_ROI);
rICT = zeros(N,N_ROI);
AREA = zeros(N,1);

% Initialize parallelization
if isempty(gcp('nocreate')); parpool; end
fprintf('Processing %d timepoints with %d-frame averaging across %d ROIs\n', N, n_avg, N_ROI);
tic;

%p = ParforProgressbar(N, 'title', 'Computing ROI ICT');
    p = ParforProgMon('Computing ROI ICT - ', N);

parfor i = 1:N
  
  % Load n sc frames and average
  SC_FRAME = read_subimage(files, -1, -1, idx(:,i));
  SC_FRAME = mean(SC_FRAME, 3)';
  
  % Calculate correlation time
  CT_FRAME = get_tc_band(SC_FRAME,5e-3,1);
  
  % Calculate size of infarct (mm^2)
  AREA(i) = length(find(CT_REF./CT_FRAME < core_threshold))/(scale^2);
  
  % Calculate SC, CT, and ICT_REL within each ROI
  for j = 1:N_ROI
    
    % Mask SC image and save mean
    SC_MASKED = ROI_NaN(:,:,j).*SC_FRAME; %#ok<PFBNS>
    SC(i,j) = nanmean(SC_MASKED(:));
    
    % Mask CT image and save mean
    CT_MASKED = ROI_NaN(:,:,j).*CT_FRAME;
    CT(i,j) = nanmean(CT_MASKED(:));
    
    % Calculate relative ICT and save
    rICT(i,j) = CT_REF_MASK(j)/CT(i,j); %#ok<PFBNS>
    
  end
  
  p.increment(); %#ok<PFBNS>
  
end

delete(p);
fprintf('Elapsed time: %.2fs (%.2fs per timepoint)\n',toc,toc/N);

% Create directory for data output
d_parts = strsplit(files(1).folder, filesep);
output = fullfile(pwd, ['rICT_' d_parts{end}]);
if ~exist(output,'dir')
  mkdir(output);
end

% Visualize
setSpeckleColorScheme();
plot(t,SC);
xlabel('Time (s)'); ylabel('Speckle Contrast (K)'); grid on;
title('Speckle Contrast within ROIs');
legend(names);
print(fullfile(output, 'ROI_SC.png'), '-dpng');

setSpeckleColorScheme();
plot(t,CT);
xlabel('Time (s)'); ylabel('\tau_C (s)'); grid on;
title('Correlation Time within ROIs');
legend(names);
print(fullfile(output, 'ROI_CT.png'), '-dpng');

setSpeckleColorScheme();
plot(t,rICT);
disp(length(rICT))
xlabel('Time (s)'); ylabel('Relative ICT'); grid on;
title('Relative ICT within ROIs');
legend(names);
print(fullfile(output, 'ROI_rICT.png'), '-dpng');

figure;
plot(t,AREA);
xlabel('Time (s)'); ylabel('Area (mm^2)'); grid on;
title(sprintf('Infarct Size (Relative ICT < %0.2f)', core_threshold));
print(fullfile(output, 'Infarct_Size.png'), '-dpng');

% Overlay ROI(s) onto First Frame
F1 = SpeckleFigure(SC_REF, sc_range);
F1.showScalebar();
F1.showROIs(ROI);
F1.savePNG(fullfile(output, 'ROI_Start.png'));

% Overlay ROI(s) onto Last Frame
F2 = SpeckleFigure(SC_END, sc_range);
F2.showScalebar();
F2.showROIs(ROI);
F2.savePNG(fullfile(output, 'ROI_End.png'));

CT_BASELINE = CT_REF_MASK;
save(fullfile(output, 'data.mat'),'SC','CT','CT_BASELINE','rICT',...
  'AREA','t','ROI','t_start','names');

fprintf('Figures and data saved in %s%s\n', output, filesep);

end
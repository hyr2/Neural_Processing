function rICT_plot_binned_reverse(file_sc,file_baselineCT,delta_T,varargin)
    

% rICT_plot_binned(file_sc,n_avg,file_sc_baseline) loads separate directory of 
% speckle contrast data for the rICT baseline.
%   file_sc = directory of CT-Data folder containing .mat files for CT data
%   file_sc_baseline = Path to directory of .mat files for baseline CT data
%                       if you used CT_binned.m OR Path to SCImage.bmp if you used rICT_avg.m
%                       
%   delta_T = Time resolution of each frame in the file_mat folder
%   ROI = Path to directory of ROI masks from Speckle Software or an array 
%     of binary masks. The mask dimensions must match that of the speckle data.
%   trials = How many trials are you processing. Useful for standard error

warning('off', 'Images:initSize:adjustingMag'); % Suppress image size warnings

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
validateFnc = @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'});
addRequired(p, 'file_baselineCT', @(x) exist(x, 'dir'));
addRequired(p,'delta_T', @isscalar);
addOptional(p, 'ROI', '');

addParameter(p, 'trials', @isscalar);
parse(p, file_sc, file_baselineCT,delta_T, varargin{:});

file_sc = p.Results.file_sc;

file_baselineCT = p.Results.file_baselineCT;
ROI = p.Results.ROI;
trials = p.Results.trials;
delta_T = p.Results.delta_T;

% Did you run CT_binned.m or rICT_avg.m
rict_on = input('If rICT was computed already, set this flag to 1.\n');


% Directory Stuff 
files = dir_sorted(fullfile(file_sc,'*.mat'));                  % CT data
folder = files(1).folder;
if not(rict_on)
    files_baseline = dir_sorted(fullfile(file_baselineCT,'*.mat')); % baseline
else
    files_baseline = dir_sorted(fullfile(file_baselineCT,'*.bmp')); % baseline
end
N_total = length(files);
output = fullfile(pwd, ['rICT_','ROI_CTavg']);
if ~exist(output,'dir')
  mkdir(output);
end




% Loading the baseline SC
% SC_BSL = loadReferenceSC(file_baselineSC);
% Load baseline CT
if not(rict_on)
    [CT_BSL, N_BSL] = read_subCT(files_baseline);
    CT_range = [0.05*mean(min(CT_BSL)),0.9*mean(max(CT_BSL))];
else
    folder_temp = files_baseline(1).folder;
    str_file = fullfile(folder_temp, files_baseline(1).name);
    CT_BSL = imread(str_file);
    CT_BSL = double(mean(CT_BSL,3))/255;  
    CT_range = [0 1];
end
% If ROI argument is not provided, prompt user to define
if ~isempty(ROI)
  if ischar(ROI)
    ROI = readROIs(ROI);
  else
    ROI = logical(ROI);
  end
  % Check dimensions
  if size(ROI, 1) ~= size(CT_BSL, 1) || size(ROI, 2) ~= size(CT_BSL, 2)
    error('ROI dimensions (%dx%d) do not match speckle data (%dx%d)', size(ROI,1), size(ROI, 2), size(CT_BSL, 1), size(CT_BSL, 2));
  end
else
  ROI = drawROIs(CT_BSL, CT_range);
end



% Introduce NaN for computational purposes
ROI_NaN = double(ROI);
ROI_NaN(ROI_NaN == 0) = NaN;

% Generate name labels for the ROIs
N_ROI = size(ROI,3);
names = generateROILabels(N_ROI);

% Calculate baseline CT values within each ROI
if not(rict_on)
    CT_REF_MASK = zeros(1,N_ROI);
    for i = 1:N_ROI
      CT_MASKED = ROI_NaN(:,:,i).*CT_BSL;
      CT_REF_MASK(i) = nanmean(CT_MASKED(:));
    end
end

% Preallocate Storage
CT = zeros(length(files),N_ROI);
rICT = zeros(length(files),N_ROI);
rICT_error = zeros(length(files),N_ROI);
ndCT = zeros(length(files),N_ROI);
ndCT_final = zeros(length(files),N_ROI);
ndCT_error_final = zeros(length(files),N_ROI);
fprintf('Processing %d time-bins with %d-trial averaging across %d ROIs\n', N_total, trials, N_ROI);
tic;

for i = [1:length(files)]
    % load CT for each time bin
    str_file = fullfile(folder, files(i).name);
    load(str_file,'avg','S','avg_ndeltaCT','S_ndeltaCT');
    % rICT
    stdE = S/sqrt(trials);      % this is the standard error of each time bin
    avg_plus = avg + stdE;      % Its the same for avg_minus as well
    % normalized change in CT
    stdE_ndCT = S_ndeltaCT/sqrt(trials);      % this is the standard error of each time bin
    avg_plus_ndCT = avg_ndeltaCT + stdE_ndCT;      % Its the same for avg_minus as well
    % ROI analysis
    for j = 1:N_ROI
        % Mask CT image
        CT_MASKED = ROI_NaN(:,:,j).*avg;
        CT_MASKED_plus = ROI_NaN(:,:,j).*avg_plus;
        CT(i,j) = mean(CT_MASKED(:),'omitnan');
        CT_error = mean(CT_MASKED_plus(:),'omitnan');
        % Repeat for normalized delta CT
        CT_MASKED_ndCT = ROI_NaN(:,:,j).*avg_ndeltaCT;
        CT_MASKED_plus_ndCT = ROI_NaN(:,:,j).*avg_plus_ndCT;
        ndCT(i,j) = mean(CT_MASKED_ndCT(:),'omitnan');
        ndCT_error = mean(CT_MASKED_plus_ndCT(:),'omitnan');
        if not(rict_on)
            % Calculate relative ICT
            rICT(i,j) = CT_REF_MASK(j)/CT(i,j); 
            CT_error =  CT_REF_MASK(j)/CT_error;
            rICT_error(i,j) = CT_error - rICT(i,j);
        else
            rICT(i,j) = CT(i,j);
            rICT_error(i,j) = CT_error - CT(i,j);
            % Repeat for normalized delta CT
            ndCT_final(i,j) = ndCT(i,j);
            ndCT_error_final(i,j) = ndCT_error - ndCT(i,j);
        end
    end
end

% predetermined time axis
           
t = [0:delta_T:delta_T*(N_total-1)];
% Overlay ROI(s) onto First Frame
F1 = SpeckleFigure(CT_BSL, CT_range);
% F1.showScalebar();
F1.showROIs(ROI);
F1.savePNG(fullfile(output, 'ROI_Start.png'));



lineProps.col = setSpeckleColorScheme();
lineProps.col = num2cell(lineProps.col,2);
lineProps.col([N_ROI+1:14]) = [];
mseb(t,rICT',rICT_error',lineProps);
xlabel('Time (s)'); ylabel('Relative ICT'); grid on;
title('Relative ICT within ROIs');
print(fullfile(output, 'ROI_rICT.png'), '-dpng');
figure;
mseb(t,ndCT_final',ndCT_error_final',lineProps);
xlabel('Time (s)'); ylabel('n\DeltaCT'); grid on;
title('Normalized \Delta CT within ROIs');
legend(names);
print(fullfile(output, 'ROI_ndCT.png'), '-dpng');
save(fullfile(output,'data.mat'),'ROI','rICT','rICT_error','ndCT_final','ndCT_error_final','t','names');

end



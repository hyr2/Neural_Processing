function rICT_Full(file_sc,file_baselineCT,delta_T,varargin)
    

% rICT_Full(file_sc,n_avg,file_sc_baseline): Generates rICT plots. Generates
% video. Draw ROIs, filter image stack and compute standard error of the 
% mean from a list of rICT images.   

% loads separate directory of 
% speckle contrast data for the rICT baseline.
%   file_sc = directory of CT-Data folder containing .mat files for CT data
%   file_sc_baseline = Path to directory of .mat files for baseline CT data
%                       if you used CT_binned.m
%                       OR Path to SCImage.bmp if you used rICT_avg_filt.m
%                       
%   delta_T = Time resolution of each frame in the file_mat folder
%   ROI = Path to directory of ROI masks from Speckle Software or an array 
%     of binary masks. The mask dimensions must match that of the speckle data.
%   trials = How many trials are you processing. Useful for standard error
%   video_range = range of rICT values for video
%   vessel_flag = Enable this to remove activation from vessels
%   stim_start_time = If this value is specified, shows peri-stimulus
%     zoomed in plot. Value is in samples, not seconds

warning('off', 'Images:initSize:adjustingMag'); % Suppress image size warnings

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
validateFnc = @(x) validateattributes(x, {'numeric'}, {'numel', 2, 'increasing'});
addRequired(p, 'file_baselineCT', @(x) exist(x, 'dir'));
addRequired(p,'delta_T', @isscalar);
addOptional(p, 'ROI', '');
addOptional(p, 'ndCT', true);
addOptional(p, 'video_range',[0.99 1.06]);
addParameter(p, 'trials', @isscalar);
addOptional(p,'vessel_flag', false, @islogical);
addOptional(p,'stim', [0,1]);
parse(p, file_sc, file_baselineCT,delta_T, varargin{:});

file_sc = p.Results.file_sc;

file_baselineCT = p.Results.file_baselineCT;
ROI = p.Results.ROI;
trials = p.Results.trials;
delta_T = p.Results.delta_T;
ndCT = p.Results.ndCT;
video_range = p.Results.video_range;
vessel_flag = p.Results.vessel_flag;
stim = p.Results.stim;

% Directory Stuff 
files = dir_sorted(fullfile(file_sc,'*.mat'));                  % CT data
folder = files(1).folder;
files_baseline = dir_sorted(fullfile(file_baselineCT,'*.sc')); % baseline
N_total = length(files);
d_parts = strsplit(file_sc, filesep);
output = fullfile(pwd, [d_parts{end} '_Full-Analysis']);
if ~exist(output,'dir')
  mkdir(output);
end

% Load baseline CT
folder_temp = files_baseline(1).folder;
str_file = fullfile(folder_temp, files_baseline(1).name);
SC = read_subimage(files_baseline);
[imh,imw] = size(SC);

% If ROI argument is not provided, prompt user to define
if ~isempty(ROI)
    load(ROI,'BW')
    BW = logical(BW);
  if ischar(ROI)
    ROI = readROIs(ROI);
  else
    ROI = logical(ROI);
  end
  % Check dimensions
  if size(ROI, 1) ~= size(SC, 1) || size(ROI, 2) ~= size(SC, 2)
    error('ROI dimensions (%dx%d) do not match speckle data (%dx%d)', size(ROI,1), size(ROI, 2), size(SC, 1), size(SC, 2));
  end
else
    ROI = drawROIs(SC, [0.02 0.35]);
  % Define craniotomy
    BW = defineCraniotomy(SC,[0.018,0.35]);
end

% Digital filter (performed in rICT_avg_fltr)
% order = 4;      % Order of the IIR filter
% fcutlow = 2;    % cutoff frequency
% Fs = 1/delta_T;        % Sampling frequency
% lowpass_filter = designfilt('lowpassiir','FilterOrder',order, ...
%     'StopbandFrequency',fcutlow,'SampleRate',Fs,'DesignMethod','cheby2');

% Image stack
rICT_final = zeros(imh,imw,N_total);        % Empty matrix of all sequences
rICT_StdE = zeros(imh,imw,N_total);
if not(ndCT)
    ndCT_final = zeros(imh,imw,N_total);
    ndCT_StdE = zeros(imh,imw,N_total);
end
for i = [1:length(files)]   % loop over sequences
    % load CT for each time bin
    str_file = fullfile(folder, files(i).name);
    load(str_file,'avg','S');
    % Storing in a 3D array
    rICT_final(:,:,i) = avg;
    rICT_StdE(:,:,i) = S/sqrt(trials);                 % standrad error of each time bin
    if not(ndCT)
        load(str_file,'avg_ndeltaCT','S_ndeltaCT');
        ndCT_final(:,:,i) = avg_ndeltaCT;
        ndCT_StdE(:,:,i) = S_ndeltaCT/sqrt(trials);        % standard error of each time bin
    end
end

% Filtering (performed in rICT_avg_fltr)
% rICT_final(isnan(rICT_final)) = -1;                 % replacing NaN with zeros
% rICT_final = permute(rICT_final,[3 2 1]);           % Row index is time
% rICT_final = filtfilt(lowpass_filter,rICT_final);   % Filtering on row index
% rICT_final = permute(rICT_final,[3 2 1]);           % Z index is time
% rICT_final(rICT_final == -1) = NaN;                 % Replacing NaNs back

% if not(ndCT)
%     ndCT_final(isnan(ndCT_final)) = -1;                 % replacing NaN with zeros
%     ndCT_final = permute(ndCT_final,[3 2 1]);           % Row index is time
%     ndCT_final = filtfilt(lowpass_filter,ndCT_final);   % Filtering on row index
%     ndCT_final = permute(ndCT_final,[3 2 1]);           % Z index is time
%     ndCT_final(ndCT_final == -1) = NaN;                 % Replacing NaNs back
% end

% Time axis
t = [0:delta_T:delta_T*(N_total-1)];

% Video of filtered data 
fps = 20;
if not(ndCT)
    Generate_video_stack(ndCT_final,fps,delta_T,output,SC,BW);
end
if vessel_flag == 1
    T = adaptthresh(SC,0.75);
    BW = imbinarize(SC,T);
    y = rICT_final .* BW;
    Generate_video_stack(y,fps,delta_T,output,SC,BW,'label','rICT','overlay_range',video_range);
else
    Generate_video_stack(rICT_final,fps,delta_T,output,SC,BW,'label','rICT','overlay_range',video_range,'show_scalebar',true);
end


% Introduce NaN for computational purposes
ROI_NaN = double(ROI);
ROI_NaN(ROI_NaN == 0) = NaN;

% Generate name labels for the ROIs
N_ROI = size(ROI,3);
names = generateROILabels(N_ROI);

% Preallocate Storage
CT = zeros(length(files),N_ROI);
rICT = zeros(length(files),N_ROI);
rICT_error = zeros(length(files),N_ROI);
if not(ndCT)
    ndCT = zeros(length(files),N_ROI);
    ndCT_final_f = zeros(length(files),N_ROI);
    ndCT_error_final = zeros(length(files),N_ROI);
end
fprintf('Processing %d time-bins with %d-trial averaging across %d ROIs\n', N_total, trials, N_ROI);
tic;

for i = [1:N_total]
    % rICT
    avg = rICT_final(:,:,i);
    stdE = rICT_StdE(:,:,i);
    avg_plus = avg + stdE;                         % Its the same for avg_minus as well
    % normalized change in CT
    if not(ndCT)
        avg_ndeltaCT = ndCT_final(:,:,i);
        stdE_ndCT = ndCT_StdE(:,:,i);
        avg_plus_ndCT = avg_ndeltaCT + stdE_ndCT;      % Its the same for avg_minus as well
    end

    % ROI analysis
    for j = 1:N_ROI
        % Mask CT image
        CT_MASKED = ROI_NaN(:,:,j).*avg;
        CT_MASKED_plus = ROI_NaN(:,:,j).*avg_plus;
        CT(i,j) = mean(CT_MASKED(:),'omitnan');
        CT_error = mean(CT_MASKED_plus(:),'omitnan');
        % Computing errors         
        rICT(i,j) = CT(i,j);
        rICT_error(i,j) = CT_error - CT(i,j);
        if not(ndCT)
            % Repeat for normalized delta CT
            CT_MASKED_ndCT = ROI_NaN(:,:,j).*avg_ndeltaCT;
            CT_MASKED_plus_ndCT = ROI_NaN(:,:,j).*avg_plus_ndCT;
            ndCT(i,j) = mean(CT_MASKED_ndCT(:),'omitnan');
            ndCT_error = mean(CT_MASKED_plus_ndCT(:),'omitnan');
            % Repeat for normalized delta CT
            ndCT_final_f(i,j) = ndCT(i,j);
            ndCT_error_final(i,j) = ndCT_error - ndCT(i,j);
        end
    end
end

% setting distorted rICT values to zero
skip_distortion = 4;            % Skip distortion effects at the end due to filtering
rICT([1:skip_distortion],:) = 1;
rICT([N_total-skip_distortion+1:N_total],:) = 1;

% Overlay ROI(s) onto First Frame
F1 = SpeckleFigure(SC, [0.02 0.38]);
F1.showScalebar();
F1.showROIs(ROI);
F1.savePNG(fullfile(output, 'ROI_Overlay.png'));
% with error bars
lineProps.col = setSpeckleColorScheme();
lineProps.col = num2cell(lineProps.col,2);
lineProps.col([N_ROI+1:14]) = [];
if trials == 1
    setSpeckleColorScheme();
    plot(t,rICT','LineWidth',1.8);
    xlabel('Time (s)'); ylabel('Relative ICT'); grid on;
    title('Relative ICT within ROIs');
    legend(names);
    saveas(gcf,fullfile(output, 'ROI_rICT_errors.svg'),'svg');
    print(fullfile(output, 'ROI_rICT_errors.png'), '-dpng');
    if stim(1) ~= 0     % Zooming in around activation area 
        zm_y = rICT([stim(1)-10:stim(1)+stim(2)+15],1);
        zm_t = t([stim(1)-10:stim(1)+stim(2)+15]);
        lineProps.col = setSpeckleColorScheme();
        lineProps.col = num2cell(lineProps.col,2);
        lineProps.col([N_ROI+1:14]) = [];
        setSpeckleColorScheme();
        plot(zm_t,zm_y,'LineWidth',2);
        title('Peri-stimulus rICT');
        ax = gca; 
        ax.FontSize = 16;
        y_lim = [min(zm_y),max(zm_y)];
        xline(t(stim(1)-1),'--','LineWidth',2,'color','g');
        xline(t(stim(1) + stim(2)),'--','LineWidth',2,'color','g');
        saveas(gcf,fullfile(output, 'PS_rICT.svg'),'svg');
        print(fullfile(output, 'PS_rICT.png'), '-dpng');
    end
else
    mseb(t,rICT',rICT_error',lineProps);
    xlabel('Time (s)'); ylabel('Relative ICT'); grid on;
    title('Relative ICT within ROIs');
    legend(names);
    saveas(gcf,fullfile(output, 'ROI_rICT_errors.svg'),'svg');
    print(fullfile(output, 'ROI_rICT_errors.png'), '-dpng')
    if stim(1) ~= 0    % Zooming in around activation area
        zm_y = rICT([stim(1)-10:stim(1)+stim(2)+15],1);
        zm_t = t([stim(1)-10:stim(1)+stim(2)+15]);
        lineProps.col = setSpeckleColorScheme();
        lineProps.col = num2cell(lineProps.col,2);
        lineProps.col([N_ROI+1:14]) = [];
        setSpeckleColorScheme();
        plot(zm_t,zm_y,'LineWidth',2);
        title('Peri-stimulus rICT');
        ax = gca; 
        ax.FontSize = 16;
        y_lim = [min(zm_y),max(zm_y)];
        xline(t(stim(1)-1),'--','LineWidth',2,'color','g');
        xline(t(stim(1) + stim(2)),'--','LineWidth',2,'color','g');
        saveas(gcf,fullfile(output, 'PS_rICT.svg'),'svg');
        print(fullfile(output, 'PS_rICT.png'), '-dpng');
    end
end
% without errorbars

if trials ~= 1
    lineProps.col = setSpeckleColorScheme();
    lineProps.col = num2cell(lineProps.col,2);
    lineProps.col([N_ROI+1:14]) = [];
    plot(t,rICT','LineWidth',1.8);
    xlabel('Time (s)'); ylabel('Relative ICT'); grid on;
    title('Relative ICT within ROIs');
    legend(names);
    saveas(gcf,fullfile(output, 'ROI_rICT.fig'));
end
% for ndCT
if not(ndCT)
    figure;
    if trials == 1
        setSpeckleColorScheme();
        plot(t,ndCT_final_f','LineWidth',1.8);
    else
        mseb(t,ndCT_final_f',ndCT_error_final',lineProps);
    end
    xlabel('Time (s)'); ylabel('n\DeltaCT'); grid on;
    title('Normalized \Delta CT within ROIs');
    legend(names);
    print(fullfile(output, 'ROI_ndCT.png'), '-dpng');
end

%% Box-whisker Plot for significance testing


if not(ndCT)
    save(fullfile(output,'ROI.mat'),'BW','ROI','rICT','rICT_error','ndCT_final_f','ndCT_error_final','t','names');
else
    save(fullfile(output,'ROI.mat'),'BW','ROI','rICT','rICT_error','t','names','N_ROI');
end
close all;
end
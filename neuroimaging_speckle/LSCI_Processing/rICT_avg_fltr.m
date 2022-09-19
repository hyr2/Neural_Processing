function rICT_avg_fltr(file_sc,len_trials,num_bin,trials,sc_range,varargin)
%rICT_avg_fltr Computes CT for each sequence and then averages along the time
%bins. Trial based averaging
% rICT_avg_fltr(file_sc,len_trials,num_bin,trials,sc_range) generates a .mat file for each time
% bin. Thus, we are averaging over a fixed number of trials: 'trials',
% This .mat file contains: 
%   1) mean and standard deviation of each time bin along trials
%   2) Count of the trials used in the processing
%
%   INPUT ARGUMENTS:
%   file_sc = Path to directory of speckle contrast files
%   len_trials = number of time bins in the experiment (len_trials)
%   num_bin = Size of each time bin (in # of seq)
%   trials = number of trials to include (IMPORTANT : specify as vector [initial trial, final trial])
%   e.g: to study the data of the 51-100 trials you would do: [51,100]
%   file_dst = Path for the destination processed data
%   delta_t = time resolution of experimental aquisition (single time bin)
%   baseline = {'F','B'}. 'F': start of trial. 'B': end of trial
%   num_BSL = number of sequences to be selected as baseline
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS

warning('off', 'Images:initSize:adjustingMag'); % Suppress image size warnings

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'len_trials', @isscalar);
addRequired(p, 'num_bin', @isscalar);
addRequired(p, 'trials',  @isvector);
addRequired(p, 'sc_range', @isvector);
addOptional(p, 'file_dst', '');
addOptional(p, 'delta_t', @isscalar);
addOptional(p, 'baseline', 'B');
addOptional(p,'num_BSL', 20);
addParameter(p, 'resolution', get(0, 'screenpixelsperinch'), @isscalar);
parse(p, file_sc, len_trials, num_bin, trials, sc_range,varargin{:});
file_sc = p.Results.file_sc;
sc_range = p.Results.sc_range;
len_trials = p.Results.len_trials;
trials = p.Results.trials;
num_bin = p.Results.num_bin;
file_dst = p.Results.file_dst;
resolution = p.Results.resolution;
baseline = p.Results.baseline;
delta_t = p.Results.delta_t;
num_BSL = p.Results.num_BSL;

if ~exist(file_dst,'dir')
  mkdir(file_dst);
end

trials_initial = trials(1);
trials = trials(2) - trials(1) + 1;

% Get list of speckle contrast files
files = dir_sorted(fullfile(file_sc, '*.sc'));
SC = read_subimage(files,-1,-1,[1:20]);
SC = mean(SC, 3)';
F = SpeckleFigure(SC, sc_range);
filename = fullfile(file_sc,'SCImage');
F.saveBMP(filename, resolution);
dim_SC = size(SC);

%%

start_trial = [1:len_trials:1000*len_trials];
start_trial = start_trial(trials_initial:trials+trials_initial-1);
% Statistics 
avg = zeros(dim_SC(1),dim_SC(2),'double');
S = zeros(dim_SC(1),dim_SC(2),'double');

% Baseline
% num_BSL = input('How many sequences do you want to set as baseline?\n');
if num_BSL > len_trials
    fprintf('Value of # of baseline sequences was too high!')
    quit force;
end
% If total number of sequences are odd, this will add the last sequence to
% the baseline such that no sequences is wasted
if mod(len_trials,2) == 1
    num_BSL = num_BSL + 1;
end

x = floor(len_trials/num_bin);

% Digital filter
order = 4;      % Order of the IIR filter
fcutlow = 2.75;    % cutoff frequency
Fs = 1/delta_t;        % Sampling frequency
lowpass_filter = designfilt('lowpassiir','FilterOrder',order, ...
    'StopbandFrequency',fcutlow,'SampleRate',Fs,'DesignMethod','cheby2');

% Specify baseline
skip_distortion = 4;            % Skip distortion effects at the start and end due to filtering
if strcmp(baseline,'F')
    iter_bsl = [skip_distortion+1:skip_distortion+1+num_BSL];       % if baseline at the start
else
    iter_bsl = [x-num_BSL:x-skip_distortion];                       % if baseline at the end
end

%% Computing CT and then finding statistics
tic;
% Big 4D CT matrix
% 4th dimension is trials
% 3rd dimension is time bins
arr_t = zeros(dim_SC(1),dim_SC(2),x,trials);
fprintf('Computing correlation times from filtered speckle contrast images... \n');
for iter_trial = 1:trials
    arr = zeros(dim_SC(1),dim_SC(2),x);
    SC = read_subimage(files,-1,-1,[start_trial(iter_trial):start_trial(iter_trial)+len_trials-1]);
%     SC = zeropad_stack(SC,skip_distortion);         % zero padding before filtering
    % Filtering each trial's stack
    SC(isnan(SC)) = 1;                 % replacing NaN with zeros
    SC = permute(SC,[3 2 1]);           % Row index is time
    SC = filtfilt(lowpass_filter,SC);   % Filtering on row index
    SC = permute(SC,[3 2 1]);           % Z index is time
%     SC(:,:,[1:skip_distortion,end-skip_distortion+1]) = [];
    for iter_seq = 1:x
        CT = get_tc_band(SC(:,:,iter_seq),5e-3,1); 
        arr(:,:,iter_seq) = CT';
    end
    arr_t(:,:,:,iter_trial) = arr;           % This array contains CTs
    clear SC arr
    fprintf('%3d %% done \n', iter_trial/trials*100);
end

% rICT 
rICT = zeros(dim_SC(1),dim_SC(2),trials);
fprintf('Computing rICT and saving .mat files... \n');
for j = 1:x
    filename = fullfile(file_dst,strcat('CT-',sprintf('%03d',j),'.mat'));
    for iter_trial = 1:trials
        ct_bsl = arr_t(:,:,iter_bsl,iter_trial);         % baseline CT
        ct_bsl = mean(ct_bsl,3,'omitnan');
        rICT(:,:,iter_trial) = ct_bsl./arr_t(:,:,j,iter_trial);
    end
    [avg, S, N_trials] = stats_images(rICT);  % computing statistics for the time bin
    save(filename, 'avg' , 'S' , 'N_trials');
    fprintf('%3d %% done \n', j/x*100);
end
fprintf('Elapsed time %.2fs \n', toc);
clear avg S rICT arr_t arr
end

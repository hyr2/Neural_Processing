function rICT_avg(file_sc,len_trials,num_bin,trials,sc_range,varargin)
%rICT_avg Computes CT for each sequence and then averages w.r.t the time
%bin. Trial based averaging
% rICT_avg(file_sc,len_trials,trials) generates a .mat file for each time
% bin. Thus, we are averaging over a fixed number of trials: 'trials',
% This .mat file contains: 
%   1) Averaged CT values pixel-by-pixel for each time bin 
%   2) mean and standard deviation of each time bin
%   3) Count of the trials used in the processing
%
%   file_sc = Path to directory of speckle contrast files
%   len_trials = number of time bins in the experiment (len_trials)
%   num_bin = Size of each time bin (in # of seq)
%   trials = number of trials to include (IMPORTANT : specify as vector [initial trial, final trial])
%   e.g: to study the data of the 51-100 trials you would do: [51,100]
%   file_dst = Path for the destination processed data
%  
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
addParameter(p, 'resolution', get(0, 'screenpixelsperinch'), @isscalar);
parse(p, file_sc, len_trials, num_bin, trials, sc_range,varargin{:});
file_sc = p.Results.file_sc;
sc_range = p.Results.sc_range;
len_trials = p.Results.len_trials;
trials = p.Results.trials;
num_bin = p.Results.num_bin;
file_dst = p.Results.file_dst;
resolution = p.Results.resolution;

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
num_BSL = input('How many sequences do you want to set as baseline?\n');
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

% automated rejection of bad trials
min_rict = 1;
max_rict = 1.1;

%% Computing CT and then finding statistics

% This for loop is used to generate the baseline matrix stack (3D array).
% 3rd dimension is the trial
fprintf('Computing baseline matrix for %d trials\n',trials)
arr_bsl = double.empty(dim_SC(1),dim_SC(2),0);  % dim 3 is trial
arr_bsl_temp = double.empty(dim_SC(1),dim_SC(2),0);
for k = 1:trials
    iter_bsl = [start_trial(k)+len_trials-num_BSL:start_trial(k)+len_trials-1];
    for iter = iter_bsl
        SC_BSL = read_subimage(files,-1,-1,[iter]);
        SC_BSL = mean(SC_BSL, 3,'omitnan')'; % you are not averaging anything here. delete?
        CT_BSL = get_tc_band(SC_BSL, 5e-3, 1);
        arr_bsl_temp = cat(3,arr_bsl_temp,CT_BSL);
    end
    arr_bsl_temp = mean(arr_bsl_temp, 3,'omitnan');     % avg of all baseline CTs
    arr_bsl = cat(3,arr_bsl,arr_bsl_temp);
end
% This for loop is over time bins
for j = 1:x
    % First for loop is for creating a list of indices
    arr = int32.empty(0,num_bin*trials);
    for i = 1:trials
        add_arr = [start_trial(i)+(j-1)*num_bin:start_trial(i)+(num_bin-1)+(j-1)*num_bin];
        arr = [arr,add_arr];
    end
    fprintf('Working on bin #%d\n',j);
    filename = fullfile(file_dst,strcat('CT-',sprintf('%03d',j),'.mat'));
    arr_rICT = double.empty(dim_SC(1),dim_SC(2),0);
    arr_ndeltaCT = double.empty(dim_SC(1),dim_SC(2),0);
    for k = 1:trials
        SC = read_subimage(files, -1, -1, arr(num_bin*k-num_bin+1:num_bin*k)); 
        SC = mean(SC, 3,'omitnan')'; % you are not averaging here. delete?
        CT = get_tc_band(SC, 5e-3, 1);  % compute CT assuming 5 ms exposure time
        % rICT
        rICT = arr_bsl(:,:,k)./CT;
        ndeltaCT = -(CT-arr_bsl(:,:,k))./arr_bsl(:,:,k);
        % Storing
        arr_rICT = cat(3,arr_rICT,rICT);    % collecting all rCT for one time bin
        arr_ndeltaCT = cat(3,arr_ndeltaCT,ndeltaCT);    % collecting all normalized change in CT for one time bin
    end
    [avg, S, N_trials] = stats_images(arr_rICT);  % computing statistics for the time bin
    [avg_ndeltaCT, S_ndeltaCT, N_trials] = stats_images(arr_ndeltaCT);
    save(filename, 'avg_ndeltaCT' , 'S_ndeltaCT' , 'avg' , 'S' , 'N_trials');
end      
end

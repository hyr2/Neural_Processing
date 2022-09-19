function CT_binned(file_sc,n_repeat,trials,varargin)
%CT_binned Computes CT for each sequence and then averages w.r.t the time
%bin. Trial based averaging
% CT_binned(file_sc,n_repeat,trials) generates a .mat file for each time
% bin. Thus, we are averaging over a fixed number of trials: 'trials',
% This .mat file contains: 
%   1) Averaged CT values pixel-by-pixel for each time bin 
%   2) mean and standard deviation of each time bin
%   3) Count of the trials used in the processing
%
%   file_sc = Path to directory of speckle contrast files
%   n_repeat = number of time bins in the experiment
%   trials = number of trials to include ("1 to trials" is default)
%   file_dst = Path for the destination processed data
%   trials_initial = use this if you want to analyze trials : [initial initial+trials]
%
% ASSUMES 5MS EXPOSURE TIME FOR INVERSE CORRELATION TIME CALCULATIONS

warning('off', 'Images:initSize:adjustingMag'); % Suppress image size warnings

p = inputParser;
addRequired(p, 'file_sc', @(x) exist(x, 'dir'));
addRequired(p, 'n_repeat', @isscalar);
addRequired(p, 'trials', @isscalar);
addOptional(p, 'file_dst', '');
addParameter(p, 'trials_initial', @isscalar);
parse(p, file_sc, n_repeat, trials, varargin{:});
file_sc = p.Results.file_sc;
n_repeat = p.Results.n_repeat;
trials = p.Results.trials;
file_dst = p.Results.file_dst;
trials_initial = p.Results.trials_initial;

if (nargin == 5)
    trials_initial = 1;
end

if ~exist(file_dst,'dir')
  mkdir(file_dst);
end

% Get list of speckle contrast files
files = dir_sorted(fullfile(file_sc, '*.sc'));
SC = read_subimage(files,-1,-1,1);
SC = mean(SC, 3)';
dim_SC = size(SC);

% Draw ROIs on the SC image



file_sc = files(1).folder;
start_trial = [1:n_repeat:1000*n_repeat];
start_trial = start_trial(trials_initial:trials+trials_initial-1);
% Statistics 
avg = zeros(dim_SC(1),dim_SC(2),'double');
S = zeros(dim_SC(1),dim_SC(2),'double');

% Computing CT and then finding statistics
for j = 1:n_repeat
        filename = fullfile(file_dst,strcat('CT-',sprintf('%03d',j),'.mat'));
        arr = int32.empty(0,trials);
        for i = 1:trials
            add_arr = [start_trial(i)+(j-1):start_trial(i)+(j-1)];
            arr = [arr,add_arr];
        end
        fprintf('Working on bin #%d\n',j);
        arr_CT = double.empty(dim_SC(1),dim_SC(2),0);
        for k = 1:trials
            SC = read_subimage(files, -1, -1, arr(k));
            SC = mean(SC, 3)';
            CT = get_tc_band(SC, 5e-3, 1);  % compute CT assuming 5 ms exposure time
            arr_CT = cat(3,arr_CT,CT);    % collecting all CT for one time bin
        end
        [avg, S, N_trials] = stats_images(arr_CT);  % computing statistics for the time bin
%         save(filename, 'arr_CT' , 'avg' , 'S' , 'N_trials');
        save(filename, 'avg' , 'S' , 'N_trials');
end
clear all;
close all;
end
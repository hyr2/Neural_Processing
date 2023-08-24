function [R_i] = GSR_z(Data_cell,g_signal_orig)
%GSR_MAIN Performs regression on signals (1D temporal). Each pixel of the
%image is considered as a time signal
%   Input Parameters:
%   Data_cell: input 1D cell array. Each element is a 2D image. Row dimension
%   of the cell array sequences in time 
%   g_signal : input global signal of the image frame as a function of time
%               (a column vector)

%   GSR is based on a weighted subtraction of the global signal. The
%   weights are proportional to the correlation between the global signal
%   time-series and the pixel i time-series. It exactly becomes the
%   correlation if the mean of pixel i time series is zero (just subtract
%   the mean from C(t)).

%   Using the math in this paper: https://doi.org/10.1016/j.neuroimage.2008.09.036
%   eq(1-3)


[X,Y] = size(Data_cell{5});
[num_trials,len_trials] = size(Data_cell);
N = num_trials * len_trials;
mat_all_trials = zeros(N,X,Y);
% mat_single_trial_subt = zeros(N,X,Y);
R_signal = zeros(len_trials,X,Y);

% Creating 3D stack of images (first dimension is time)
for iter_n = 1:N
    mat_all_trials(iter_n,:,:) = Data_cell{iter_n}(:,:);
end
clear Data_cell

mat_all_trials = reshape(mat_all_trials,N,[]);
mat_all_trials = mat_all_trials - mean(mat_all_trials,1); % mean removed

beta_i = (g_signal_orig.'*g_signal_orig)^(-1) * (g_signal_orig.' * mat_all_trials);       % for all pixels (Beta is a row vector)

fprintf('Max beta value %f and Min beta value %f', max(beta_i) ,min(beta_i));

R_i = mat_all_trials.' - ( beta_i.' * g_signal_orig.');
R_i = R_i.';

R_i = reshape(R_i,4500,X,Y);


end


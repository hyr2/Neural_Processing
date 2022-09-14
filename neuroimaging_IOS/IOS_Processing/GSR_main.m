function [R_signal] = GSR_main(Data_cell,g_signal_orig)
%GSR_MAIN Performs regression on signals (1D temporal). Each pixel of the
%image is considered as a time signal
%   Input Parameters:
%   Data_cell: input 1D cell array. Each element is a 2D image. Row dimension
%   of the cell array sequences in time 
%   g_signal : input global signal of the image frame as a function of time

%   GSR is based on a weighted subtraction of the global signal. The
%   weights are proportional to the correlation between the global signal
%   time-series and the pixel i time-series. It exactly becomes the
%   correlation if the mean of pixel i time series is zero (just subtract
%   the mean from C(t)).

%   Using the math in this paper: https://doi.org/10.1016/j.neuroimage.2008.09.036
%   eq(1-3)


[X,Y] = size(Data_cell{5});
[~,len_trials] = size(Data_cell);
mat_single_trial = zeros(len_trials,X,Y);
mat_single_trial_subt = zeros(len_trials,X,Y);
R_signal = zeros(len_trials,X,Y);

% Creating 3D stack of images (first dimension is time)
for iter_seq = 1:len_trials
    mat_single_trial(iter_seq,:,:) = Data_cell{iter_seq}(:,:);
end
clear Data_cell

% Mean subtracted (zero mean) [only for the computation of Beta correlation parameters]
mat_single_trial_subt = mat_single_trial - mean(mat_single_trial,1);
g_signal = mean(mat_single_trial_subt,[2,3]);
g_signal = g_signal';

% vector G (global signal column vector)
G = pinv(g_signal' * g_signal) * g_signal';     
% We use pinv to find the pseudo inverse of the matrix since there is no
% guarantee that (g_signal' * g_signal) is full rank.
% Learn about pseudo inverse here: https://www.youtube.com/watch?v=PjeOmOz9jSY
G_3d = repmat(G,[1,X,Y]);
% Taking the scalar product (inner product)
Beta = dot(G_3d,mat_single_trial_subt,1);
Beta = reshape(Beta,X,Y);

% Computing the regression signal
for iter_seq = 1:len_trials
    R_signal(iter_seq,:,:) = reshape(mat_single_trial(iter_seq,:,:),[X,Y]) - g_signal_orig(iter_seq) * Beta;
end
end


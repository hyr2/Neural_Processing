function [mat_all_trial] = spatial_gauss_filt(Data_cell,sig)
%SPATIAL_GAUSS_FILT Summary of this function goes here
%   Data_cell is a 2D cell array. Each element of the cell array is a 2D
%   image. The first dimension of the cell array is the trial dimension and
%   the second dimension is the time dimension

[num_trials, len_trials] = size(Data_cell);
[X,Y] = size(Data_cell{1,1});
mat_all_trial = zeros(num_trials,len_trials,X,Y);
mat_single_trial = zeros(len_trials,X,Y);

for iter_trial = 1:num_trials    
    for iter_seq = 1:len_trials
        mat_single_trial(iter_seq,:,:) = Data_cell{iter_trial,iter_seq};
    end
    % Apply 2D gaussian filter on individual frames here:
    mat_all_trial(iter_trial,:,:,:) = twoD_gauss(mat_single_trial,sig);
end

end
